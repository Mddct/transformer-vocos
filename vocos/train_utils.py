import math
import os

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from wenet.utils.mask import make_pad_mask

from vocos.dataset import init_dataset_and_dataloader
from vocos.discriminators import (SequenceMultiPeriodDiscriminator,
                                  SequenceMultiResolutionDiscriminator)
from vocos.loss import (MelSpecReconstructionLoss, compute_discriminator_loss,
                        compute_feature_matching_loss, compute_generator_loss)
from vocos.model import ISTFTHead, Transformer
from vocos.utils import (MelSpectrogram, get_cosine_schedule_with_warmup,
                         init_distributed)


class VocosTrainModel(torch.nn.Module):

    def __init__(self, config):
        self.feature_extractor = MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_size,
            n_mels=config.n_mels,
            padding='center',
        )
        self.backbone = Transformer(config)
        self.head = ISTFTHead(config)

    def forward(self, wav: torch.Tensor, wav_lens: torch.Tensor):
        padding = make_pad_mask(wav_lens)
        mels, mels_padding = self.feature_extractor(wav, padding)
        mels_masks = ~mels_padding
        x, mask = self.backbone(mels.transpose(1, 2), mels_masks)
        wav_g, wav_g_mask = self.head(x, mask.squeeze(1))
        wav_g = wav_g * wav_g_mask

        return wav_g, wav_g_mask


class VocosState:

    def __init__(
        self,
        config,
    ):

        init_distributed(config)
        model = VocosTrainModel(config)
        model.cuda()
        self.config = config
        self.model = torch.nn.parallel.DistributedDataParallel(model)
        self.device = config.device

        self.multiperioddisc = torch.nn.parallel.DistributedDataParallel(
            SequenceMultiPeriodDiscriminator().cuda())
        self.multiresddisc = torch.nn.parallel.DistributedDataParallel(
            SequenceMultiResolutionDiscriminator().cuda())

        self.melspec_loss = torch.nn.parallel.DistributedDataParallel(
            MelSpecReconstructionLoss(sample_rate=config.sample_rate).cuda())

        self.sample_rate = config.sample_rate
        self.learning_rate = config.learning_rate
        self.warmup_steps = config.warmup_steps
        self.mel_loss_coeff = config.mel_loss_coeff
        self.base_mel_coeff = config.mel_loss_coeff
        self.mrd_loss_coeff = config.mrd_loss_coeff
        self.pretrain_mel_steps = config.pretrain_mel_steps
        self.decay_mel_coeff = config.decay_mel_coeff

        self.max_steps = config.max_train_steps
        self.dataloader, _ = init_dataset_and_dataloader(
            config.train_data,
            config.per_device_batch_size,
            config.num_workers,
            config.prefetch,
            True,
            self.max_steps,
            sample_rate=config.sample_rate,
            seed=config.seed)
        # self.evaluate_utmos = config.evaluate_utmos
        # self.evaluate_pesq = config.evaluate_pesq
        # self.evaluate_periodicty = config.evaluate_periodicty
        # TODO: resume from optimizer step
        self.step = 0

        # TODO: user clu async torch writer
        self.writer = SummaryWriter(config.tensorboard_dir)

        # Optimizers
        self.opt_disc = optim.AdamW(
            list(self.multiperioddisc.parameters()) +
            list(self.multiresddisc.parameters()),
            lr=self.learning_rate,
            betas=(0.8, 0.9),
        )
        self.opt_gen = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.8, 0.9),
        )

        # Schedulers
        self.scheduler_disc = get_cosine_schedule_with_warmup(
            self.opt_disc, self.warmup_steps, self.max_steps // 2)
        self.scheduler_gen = get_cosine_schedule_with_warmup(
            self.opt_gen, self.warmup_steps, self.max_steps // 2)

    def train_step(self, batch, device):
        wav, wav_lens = batch['wavs'].to(device), batch['wavs_lens'].to(device)
        self.opt_gen.zero_grad()
        log_str = ''
        if self.train_discriminator:
            self.opt_disc.zero_grad()
            with torch.no_grad():
                wav_g, wavg_mask = self.model(wav, wav_lens)
                wav = wav[:, :wav_g.shape[1]]
                wav = wav * wavg_mask

            real_score_mp, real_score_mp_masks, _, _ = self.multiperioddisc(
                wav, wavg_mask)
            gen_score_mp, _, _, _ = self.multiperioddisc(
                wav_g.detach(), wavg_mask)

            real_score_mrd, real_score_mrd_masks, _, _ = self.multiresddisc(
                wav, wavg_mask)
            gen_score_mrd, _, _, _ = self.multiresddisc(
                wav_g.detach(), wavg_mask)

            loss_mp, _, _ = compute_discriminator_loss(real_score_mp,
                                                       gen_score_mp,
                                                       real_score_mp_masks)
            loss_mrd, _, _ = compute_discriminator_loss(
                real_score_mrd, gen_score_mrd, real_score_mrd_masks)
            disc_loss = loss_mp + self.mrd_loss_coeff * loss_mrd

            disc_loss.backward()
            self.opt_disc.step()
            self.scheduler_disc.step()
            # TODO: integrate simple-trainer
            self.writer.add_scalar("discriminator/total", disc_loss, self.step)
            self.writer.add_scalar("discriminator/multi_period_loss", loss_mp,
                                   self.step)
            self.writer.add_scalar("discriminator/multi_res_loss", loss_mrd,
                                   self.step)
            log_str += f'step_{self.step}: loss_disc: {disc_loss} loss_mpd: {loss_mp} loss_mrd: {loss_mrd}'
        wav_g, wavg_mask = self.model(wav, wav_lens)

        wav = wav[:, :wav_g.shape[1]]
        wav = wav * wavg_mask
        mel_loss = self.melspec_loss(wav_g, wav, wavg_mask)
        gen_loss = mel_loss * self.mel_loss_coeff

        gen_score_mp, gen_score_mp_mask, fmap_gs_mp, fmap_gs_mp_mask = self.multiperioddisc(
            wav_g, wavg_mask)
        real_score_mp, _, fmap_rs_mp, _ = self.multiperioddisc(wav, wavg_mask)

        gen_score_mrd, gen_score_mrd_mask, fmap_gs_mrd, fmaps_gs_mrd_mask = self.multiresddisc(
            wav_g, wavg_mask)
        real_score_mrd, _, fmap_rs_mrd, _ = self.multiresddisc(wav, wavg_mask)

        loss_gen_mp, _ = compute_generator_loss(gen_score_mp,
                                                gen_score_mp_mask)
        loss_gen_mrd, _ = compute_generator_loss(gen_score_mrd,
                                                 gen_score_mrd_mask)
        loss_fm_mp = compute_feature_matching_loss(fmap_rs_mp, fmap_gs_mp,
                                                   fmap_gs_mp_mask)
        loss_fm_mrd = compute_feature_matching_loss(
            fmap_rs_mrd, fmap_gs_mrd, [[fmaps_gs_mrd_mask[i]] * len(gs)
                                       for i, gs in enumerate(fmap_gs_mrd)])

        gen_loss += loss_gen_mp + self.mrd_loss_coeff * loss_gen_mrd + loss_fm_mp + self.mrd_loss_coeff * loss_fm_mrd

        self.writer.add_scalar("generator/multi_period_loss", loss_gen_mp,
                               self.step)
        self.writer.add_scalar("generator/multi_res_loss", loss_gen_mrd,
                               self.step)
        self.writer.add_scalar("generator/feature_matching_mp", loss_fm_mp,
                               self.step)
        self.writer.add_scalar("generator/feature_matching_mrd", loss_fm_mrd,
                               self.step)
        self.writer.add_scalar("generator/total_loss", gen_loss, self.step)
        self.writer.add_scalar("generator/mel_loss", mel_loss)

        gen_loss.backward()
        self.opt_gen.step()
        self.scheduler_gen.step()

        log_str += f'loss_gen {gen_loss} mel_loss {mel_loss}'

        self.step += 1
        if self.step >= self.pretrain_mel_steps:
            self.train_discriminator = True

        if self.decay_mel_coeff:
            self.mel_loss_coeff = self.base_mel_coeff * max(
                0.0, 0.5 * (1.0 + math.cos(math.pi *
                                           (self.step / self.max_steps))))
        if (self.step + 1) % self.config.log_interval:
            print(log_str)

    def train(self):
        for (i, batch) in enumerate(self.dataloader):
            self.train_step(batch, self.config.device)
            if (self.step + 1) % self.config.checkpoint_every_steps == 0:
                self.save()
            if self.step >= self.max_steps:
                print("Training complete.")
                return

    def save(self):
        checkpoint_dir = os.path.join(self.config.model_dir,
                                      f'step_{self.step}')
        os.makedirs(checkpoint_dir)

        model_state_dict = self.model.module.state_dict()
        torch.save(model_state_dict, os.path.join('checkpoint_dir',
                                                  'model.pt'))
        mpd_state_dict = self.multiperioddisc.module.state_dict()
        torch.save(mpd_state_dict, os.path.join('checkpoint_dir', 'mpd.pt'))
        mrd_state_dict = self.multiresddisc.module.state_dict()
        torch.save(mrd_state_dict, os.path.join('checkpoint_dir', 'mrd.pt'))

        opt_disc_state_dict = self.opt_disc.state_dict()
        torch.save(opt_disc_state_dict,
                   os.path.join('checkpoint_dir', 'opt_disc.pt'))
        opt_gen_state_dict = self.opt_gen.state_dict()
        torch.save(opt_gen_state_dict,
                   os.path.join('checkpoint_dir', 'opt_gen.pt'))

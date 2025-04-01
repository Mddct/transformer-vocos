import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.optim as optim
from tensorboard import summarywriter
from wenet.utils.mask import make_pad_mask

from vocos.discriminators import (SequenceMultiPeriodDiscriminator,
                                  SequenceMultiResolutionDiscriminator)
from vocos.loss import (MelSpecReconstructionLoss, compute_discriminator_loss,
                        compute_feature_matching_loss, compute_generatorl_oss)
from vocos.model import VocosModel
from vocos.utils import (MelSpectrogram, get_cosine_schedule_with_warmup,
                         init_distributed)


class VocosState:

    def __init__(
        self,
        config,
    ):

        init_distributed(config)
        model = VocosModel(config)
        model.cuda()
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

        # self.evaluate_utmos = config.evaluate_utmos
        # self.evaluate_pesq = config.evaluate_pesq
        # self.evaluate_periodicty = config.evaluate_periodicty

        self.train_discriminator = True
        self.global_step = 0
        self.max_steps = config.max_train_steps

        # TODO: user clu async torch writer
        self.writer = summarywriter(config.tensorboard_dir)

        # Optimizers
        self.opt_disc = optim.AdamW(
            list(self.multiperioddisc.parameters()) +
            list(self.multiresddisc.parameters()),
            lr=self.learning_rate,
            betas=(0.8, 0.9),
        )
        self.opt_gen = optim.AdamW(
            list(self.feature_extractor.parameters()) +
            list(self.backbone.parameters()) + list(self.head.parameters()),
            lr=self.learning_rate,
            betas=(0.8, 0.9),
        )

        # Schedulers
        self.scheduler_disc = get_cosine_schedule_with_warmup(
            self.opt_disc, self.warmup_steps, self.max_steps // 2)
        self.scheduler_gen = get_cosine_schedule_with_warmup(
            self.opt_gen, self.warmup_steps, self.max_steps // 2)

    def train_step(self, batch, device):
        wav, wav_lens = batch['wav'].to(device), batch['wav_lens'].to(device)
        self.opt_gen.zero_grad()
        if self.train_discriminator:
            self.opt_disc.zero_grad()
            with torch.no_grad():
                wav_g, wav_mask = self(wav, wav_lens)
                wav = wav[:, :wav_g.shape[1]]
                wav = wav * wavg_mask

            real_score_mp, real_score_mp_masks, _, _ = self.multiperioddisc(
                wav, wav_mask)
            gen_score_mp, _, _, _ = self.multiperioddisc(
                wav_g.detach(), wav_mask)

            real_score_mrd, real_score_mrd_masks, _, _ = self.multiresddisc(
                wav, wav_mask)
            gen_score_mrd, _, _, _ = self.multiresddisc(
                wav_g.detach(), wav_mask)

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

        wav_g, wav_mask = self(wav, wav_lens)

        wav = wav[:, :wav_g.shape[1]]
        wav = wav * wavg_mask
        mel_loss = self.melspec_loss(wav_g, wav, wav_mask)
        gen_loss = mel_loss * self.mel_loss_coeff

        gen_score_mp, gen_score_mp_mask, fmap_gs_mp, fmap_gs_mp_mask = self.multiperioddisc(
            wav_g, wav_mask)
        real_score_mp, _, fmap_rs_mp, _ = self.multiperioddisc(wav, wav_mask)

        gen_score_mrd, gen_score_mrd_mask, fmap_gs_mrd, fmaps_gs_mrd_mask = self.multiresddisc(
            wav_g, wav_mask)
        real_score_mrd, _, fmap_rs_mrd, _ = self.multiresddisc(wav, wav_mask)

        loss_gen_mp, _ = compute_generatorl_oss(gen_score_mp,
                                                gen_score_mp_mask)
        loss_gen_mrd, _ = compute_generatorl_oss(gen_score_mrd,
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

        self.step += 1
        if self.step >= self.pretrain_mel_steps:
            self.train_discriminator = True

        if self.decay_mel_coeff:
            self.mel_loss_coeff = self.base_mel_coeff * max(
                0.0, 0.5 * (1.0 + math.cos(math.pi *
                                           (self.step / self.max_steps))))

    def train(self):
        for (i, batch) in enumerate(self.dataloader):
            self.train_step(batch, config.device)
            if (self.step + 1) % config.save_interval == 0:
                self.save()
            if self.global_step >= self.max_steps:
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
        torch.save(model_state_dict, os.path.join('checkpoint_dir', 'mpd.pt'))
        mrd_state_dict = self.multiresddisc.module.state_dict()
        torch.save(model_state_dict, os.path.join('checkpoint_dir', 'mrd.pt'))

        opt_disc_state_dict = self.opt_disc.state_dict()
        torch.save(model_state_dict,
                   os.path.join('checkpoint_dir', 'opt_disc.pt'))
        opt_gen_state_dict = self.opt_gen.state_dict()
        torch.save(model_state_dict,
                   os.path.join('checkpoint_dir', 'opt_gen.pt'))

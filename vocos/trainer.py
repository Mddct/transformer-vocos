import math

import torch
import torch.optim as optim
from wenet.utils.mask import make_pad_mask

from vocos.discriminators import (MultiPeriodDiscriminator,
                                  MultiResolutionDiscriminator)
from vocos.loss import (DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss,
                        MelSpecReconstructionLoss)
from vocos.model import ISTFTHead, Transformer
from vocos.utils import MelSpectrogram, get_cosine_schedule_with_warmup


class VocosStates:

    def __init__(
        self,
        config,
    ):

        self.feature_extractor = MelSpectrogram(config)
        self.backbone = Transformer(config)
        self.head = ISTFTHead(config)

        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresddisc = MultiResolutionDiscriminator()

        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(
            sample_rate=config.sample_rate)

        self.sample_rate = config.sample_rate
        self.learning_rate = config.learning_rate
        self.warmup_steps = config.warmup_steps
        self.mel_loss_coeff = config.mel_loss_coeff
        self.base_mel_coeff = config.mel_loss_coeff
        self.mrd_loss_coeff = config.mrd_loss_coeff
        self.pretrain_mel_steps = config.pretrain_mel_steps
        self.decay_mel_coeff = config.decay_mel_coeff

        self.evaluate_utmos = config.evaluate_utmos
        self.evaluate_pesq = config.evaluate_pesq
        self.evaluate_periodicty = config.evaluate_periodicty

        self.train_discriminator = False
        self.global_step = 0
        self.max_steps = config.max_train_steps

        # TODO: user clu async torch writer
        # self.writer = SummaryWriter(log_dir)

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

    def __call__(self, wav: torch.Tensor, wav_lens: torch.Tensor):
        padding = make_pad_mask(wav_lens)

        mels, mels_padding = self.feature_extractor(wav, padding)

        x, _ = self.backbone(mels.transpose(1, 2), mels_padding)
        x = x.transpose(1, 2)
        wav_g = self.head(x)
        wav_g = wav_g * (~padding)
        return wav_g

    def train_step(self, batch, device):
        wav, wav_lens = batch['wav'].to(device), batch['wav_lens'].to(device)
        self.opt_gen.zero_grad()

        if self.train_discriminator:
            self.opt_disc.zero_grad()
            with torch.no_grad():
                wav_g = self(wav, wav_lens)
            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(
                wav, wav_g)
            real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(
                wav, wav_g)
            loss_mp, _, _ = self.disc_loss(real_score_mp, gen_score_mp)
            loss_mrd, _, _ = self.disc_loss(real_score_mrd, gen_score_mrd)
            disc_loss = loss_mp + self.mrd_loss_coeff * loss_mrd

            disc_loss.backward()
            self.opt_disc.step()
            self.scheduler_disc.step()
            # self.writer.add_scalar("Loss/Discriminator", disc_loss.item(),
            #                        self.global_step)

        wav_g = self(wav, wav_lens)
        mel_loss = self.melspec_loss(wav_g, wav)
        gen_loss = mel_loss * self.mel_loss_coeff

        if self.train_discriminator:
            _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
                wav, wav_g)
            _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
                wav, wav_g)

            loss_gen_mp, _ = self.gen_loss(gen_score_mp)
            loss_gen_mrd, _ = self.gen_loss(gen_score_mrd)
            loss_fm_mp = self.feat_matching_loss(fmap_rs_mp, fmap_gs_mp)
            loss_fm_mrd = self.feat_matching_loss(fmap_rs_mrd, fmap_gs_mrd)

            gen_loss += loss_gen_mp + self.mrd_loss_coeff * loss_gen_mrd + loss_fm_mp + self.mrd_loss_coeff * loss_fm_mrd

        gen_loss.backward()
        self.opt_gen.step()
        self.scheduler_gen.step()

        # self.writer.add_scalar("Loss/Generator", gen_loss.item(),
        #                        self.global_step)
        # self.writer.add_scalar("Loss/Mel", mel_loss.item(), self.global_step)

        self.global_step += 1
        if self.global_step >= self.pretrain_mel_steps:
            self.train_discriminator = True

        if self.decay_mel_coeff:
            self.mel_loss_coeff = self.base_mel_coeff * max(
                0.0, 0.5 *
                (1.0 + math.cos(math.pi *
                                (self.global_step / self.max_steps))))

    def fit(self, device):
        for (i, batch) in enumerate(dataloader):
            self.train_step(batch, device)
            if self.global_step >= self.max_steps:
                print("Training complete.")
                return

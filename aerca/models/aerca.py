from typing import *
import os
from tqdm import tqdm
import logging
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from torch import Tensor
import torch.nn as nn
from sklearn.metrics import f1_score
from models.senn import SENNGC
from utils.utils import (
    compute_kl_divergence,
    sliding_window_view_torch,
    eval_causal_structure,
    eval_causal_structure_binary,
    pot,
    topk,
    topk_at_step,
)
Array = np.ndarray


class AERCA(nn.Module):
    def __init__(
        self,
        num_vars: int,
        hidden_layer_size: int,
        num_hidden_layers: int,
        device: torch.device,
        window_size: int,
        stride: int = 1,
        encoder_alpha: float = 0.5,
        decoder_alpha: float = 0.5,
        encoder_gamma: float = 0.5,
        decoder_gamma: float = 0.5,
        encoder_lambda: float = 0.5,
        decoder_lambda: float = 0.5,
        beta: float = 0.5,
        lr: float = 1e-4,
        epochs: int = 100,
        recon_threshold: float = 0.95,
        data_name: str = 'ld',
        causal_quantile: float = 0.80,
        root_cause_threshold_encoder: float = 0.95,
        root_cause_threshold_decoder: float = 0.95,
        initial_z_score: float = 3.0,
        risk: float = 1e-2,
        initial_level: float = 0.98,
        num_candidates: int = 100,
    ) -> None:
        super().__init__()
        self.encoder = SENNGC(
            num_vars=num_vars,
            order=window_size,
            hidden_layer_size=hidden_layer_size,
            num_hidden_layers=num_hidden_layers,
            device=device,
        )
        self.decoder = SENNGC(
            num_vars=num_vars,
            order=window_size,
            hidden_layer_size=hidden_layer_size,
            num_hidden_layers=num_hidden_layers,
            device=device,
        )
        self.decoder_prev = SENNGC(
            num_vars=num_vars,
            order=window_size,
            hidden_layer_size=hidden_layer_size,
            num_hidden_layers=num_hidden_layers,
            device=device,
        )
        
        self.device = device
        self.num_vars = num_vars
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.window_size = window_size
        self.stride = stride
        self.encoder_alpha = encoder_alpha
        self.decoder_alpha = decoder_alpha
        self.encoder_gamma = encoder_gamma
        self.decoder_gamma = decoder_gamma
        self.encoder_lambda = encoder_lambda
        self.decoder_lambda = decoder_lambda
        self.beta = beta
        
        self.lr = lr
        self.epochs = epochs
        self.recon_threshold = recon_threshold
        self.root_cause_threshold_encoder = root_cause_threshold_encoder
        self.root_cause_threshold_decoder = root_cause_threshold_decoder
        self.initial_z_score = initial_z_score
        self.mse_loss = nn.MSELoss()
        self.mse_loss_wo_reduction = nn.MSELoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.decoder_prev.to(self.device)
        self.model_name = f'AERCA_{data_name}_ws_{window_size}_stride_{stride}_enc_alpha_{encoder_alpha}_dec_alpha_{decoder_alpha}_enc_gamma_{encoder_gamma}_dec_gamma_{decoder_gamma}_enc_lambda_{encoder_lambda}_dec_lambda_{decoder_lambda}_beta_{beta}_lr_{lr}_epochs_{epochs}_hidden_layer_size_{hidden_layer_size}_num_hidden_layers_{num_hidden_layers}'
        self.causal_quantile = causal_quantile
        self.risk = risk
        self.initial_leval = initial_level
        self.num_candidates = num_candidates
        
        self.save_dir = os.path.join(os.getcwd(), 'saved_models')
        os.makedirs(self.save_dir, exist_ok=True)

    def _log_and_print(
        self,
        msg: Any,
        *args,
    ) -> None:
        """Helper method to log and print testing results"""
        final_msg = msg.format(*args) if args else msg
        print(final_msg)

    def _sparsity_loss(
        self,
        coeffs: Tensor,
        alpha: float,
    ) -> Tensor:
        norm2 = torch.mean(torch.norm(input=coeffs, dim=1, p=2))
        norm1 = torch.mean(torch.norm(input=coeffs, dim=1, p=1))
        
        return (1 - alpha) * norm2 + alpha * norm1
    
    def _smoothness_loss(self, coeffs: Tensor) -> Tensor:
        return torch.norm(coeffs[:, 1:, :, :] - coeffs[:, :-1, :, :], dim=1).mean()
    
    def encoding(self, xs: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        windows = sliding_window_view(xs, (self.window_size+1, self.num_vars))[:, 0, :, :]
        winds = windows[:, :-1, :]
        nexts = windows[:, -1, :]
        winds = torch.tensor(winds).float().to(self.device)
        nexts = torch.tensor(nexts).float().to(self.device)
        preds, coeffs = self.encoder(winds)
        us = preds - nexts
        return us, coeffs, nexts[self.window_size:], winds[:-self.window_size]
    
    def decoding(
        self,
        us: Tensor,
        winds: Tensor,
        add_u: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        u_windows = sliding_window_view_torch(
            x=us,
            window_size=self.window_size+1
        )
        u_winds = u_windows[:, :-1, :]
        u_next = u_windows[:, -1, :]
        preds, coeffs = self.decoder(u_winds)
        prev_preds, prev_coeffs = self.decoder_prev(winds)

        if add_u:
            nexts_hat = preds + u_next + prev_preds
        else:
            nexts_hat = preds + prev_preds

        return nexts_hat, coeffs, prev_coeffs
    
    def forward(
        self,
        x: Tensor,
        add_u: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        us, encoder_coeffs, nexts, winds = self.encoding(x)
        kl_div = compute_kl_divergence(us=us, device=self.device)
        nexts_hat, decoder_coeffs, prev_coeffs = self.decoding(
            us=us,
            winds=winds,
            add_u=add_u,
        )
        return nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us
    
    def _training_step(
        self,
        x: Tensor,
        add_u: bool = True,
    ) -> Tensor:
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us = self.forward(x, add_u=add_u)
        loss_recon = self.mse_loss(nexts_hat, nexts)
        logging.info(f'Reconstruction loss: {loss_recon.item()}')

        loss_encoder_coeffs = self._sparsity_loss(coeffs=encoder_coeffs, alpha=self.encoder_alpha)
        logging.info(f'Encoder coeffs loss: {loss_encoder_coeffs.item()}')

        loss_decoder_coeffs = self._sparsity_loss(coeffs=decoder_coeffs, alpha=self.decoder_alpha)
        logging.info(f'Decoder coeffs loss: {loss_decoder_coeffs.item()}')

        loss_prev_coeffs = self._sparsity_loss(coeffs=prev_coeffs, alpha=self.decoder_alpha)
        logging.info(f'Prev coeffs loss: {loss_prev_coeffs.item()}')

        loss_encoder_smooth = self._smoothness_loss(coeffs=encoder_coeffs)
        logging.info(f'Encoder smooth loss: {loss_encoder_smooth.item()}')

        loss_decoder_smooth = self._smoothness_loss(coeffs=decoder_coeffs)
        logging.info(f'Decoder smooth loss: {loss_decoder_smooth.item()}')

        loss_prev_smooth = self._smoothness_loss(coeffs=prev_coeffs)
        logging.info(f'Prev smooth loss: {loss_prev_smooth.item()}')

        loss_kl = kl_div
        logging.info(f'KL loss: {loss_kl.item()}')

        loss = (
            loss_recon
            + self.encoder_lambda * loss_encoder_coeffs
            + self.decoder_lambda * (loss_decoder_coeffs + loss_prev_coeffs)
            + self.encoder_gamma * loss_encoder_smooth
            + self.decoder_gamma * (loss_decoder_smooth + loss_prev_smooth)
            + self.beta * loss_kl
        )
        logging.info(f'Total loss: {loss.item()}')

        return loss
    
    def _training(self, xs: Tensor) -> None:
        
        if len(xs) == 1:
            xs_train = xs[:, :int(0.8 * len(xs[0]))]
            xs_val = xs[:, int(0.8 * len(xs[0])):]
        else:
            xs_train = xs[:int(0.8 * len(xs))]
            xs_val = xs[int(0.8 * len(xs)):]
        
        best_val_loss = np.inf
        count = 0

        for epoch in tqdm(range(self.epochs), desc=f'Epoch'):
            count += 1
            epoch_loss = 0.0
            self.train()
            for x in xs_train:
                self.optimizer.zero_grad()
                loss = self._training_step(x)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            logging.info(f'Epoch {epoch + 1}/{self.epochs}')
            logging.info(f'Epoch training loss: {epoch_loss}')
            logging.info('-----------------------------')
            
            epoch_val_loss = 0.0
            self.eval()
            with torch.no_grad():
                for x in xs_val:
                    loss = self._training_step(x)
                    epoch_val_loss += loss
            logging.info(f'Epoch val loss: {epoch_val_loss}')
            logging.info('-----------------------------')
        
            if epoch_val_loss < best_val_loss:
                count = 0
                logging.info(f'Saving model at epoch {epoch + 1}')
                logging.info(f'Saving model name: {self.model_name}.pt')
                
                best_val_loss = epoch_val_loss
                torch.save(
                    obj=self.state_dict(),
                    f=os.path.join(self.save_dir, f'{self.model_name}.pt'),
                )

            if count >= 20:
                print('Early stopping')
                break
        
        self.load_state_dict(
            state_dict=torch.load(
                os.path.join(self.save_dir, f'{self.model_name}.pt'),
                map_location=self.device,
            )
        )
        logging.info('Traiing complete')
        self._get_recon_threshold(xs=xs_val)
        self._get_root_cause_threshold_encoder(xs=xs_val)
        self._get_root_cause_threshold_decoder(xs=xs_val)
        
        return None
    
    def _testing_step(
        self,
        x: Tensor,
        label: Optional[Tensor] = None,
        add_u: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us = self.forward(x=x, add_u=add_u)

        if label is not None:
            preprocessed_label = sliding_window_view(
                label, (self.window_size+1, self.num_vars)
            )[self.window_size:, 0, :-1, :]
        else:
            preprocessed_label = None
        
        loss_recon = self.mse_loss(nexts_hat, nexts)
        logging.info(f'Reconstruction loss: {loss_recon.item()}')

        loss_encoder_coeffs = self._sparsity_loss(coeffs=encoder_coeffs, alpha=self.encoder_alpha)
        logging.info(f'Encoder coeffs loss: {loss_encoder_coeffs.item()}')

        loss_decoder_coeffs = self._sparsity_loss(coeffs=decoder_coeffs, alpha=self.decoder_alpha)
        logging.info(f'Decoder coeffs loss: {loss_decoder_coeffs.item()}')

        loss_prev_coeffs = self._sparsity_loss(coeffs=prev_coeffs, alpha=self.decoder_alpha)
        logging.info(f'Prev coeffs loss: {loss_prev_coeffs.item()}')

        loss_encoder_smooth = self._smoothness_loss(coeffs=encoder_coeffs)
        logging.info(f'Encoder smooth loss: {loss_encoder_smooth.item()}')

        loss_decoder_smooth = self._smoothness_loss(coeffs=decoder_coeffs)
        logging.info(f'Decoder smooth loss: {loss_decoder_smooth.item()}')

        loss_prev_smooth = self._smoothness_loss(coeffs=prev_coeffs)
        logging.info(f'Prev smooth loss: {loss_prev_smooth}')

        loss_kl = kl_div
        logging.info(f'KL loss: {loss_kl.item()}')

        loss = (
            loss_recon
            + self.encoder_lambda * loss_encoder_coeffs
            + self.decoder_lambda * (loss_decoder_coeffs + loss_prev_coeffs)
            + self.encoder_gamma * loss_encoder_smooth
            + self.decoder_gamma * (loss_decoder_smooth + loss_prev_smooth)
            + self.beta * kl_div
        )
        logging.info(f'Total loss: {loss.item()}')

        return loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, preprocessed_label, us
    
    def _get_recon_threshold(self, xs: Tensor) -> None:
        self.eval()
        losses_list = []

        with torch.no_grad():
            for x in xs:
                _, nexts_hat, nexts, _, _, _, _, _ = self._testing_step(x=x, add_u=False)
                loss_arr = self.mse_loss_wo_reduction(nexts_hat, nexts).cpu().numpy().ravel()
                losses_list.append(loss_arr)
        recon_losses = np.concatenate(losses_list)
        self.recon_threshold_value = np.quantile(a=recon_losses, q=self.recon_threshold)
        self.recon_mean = np.mean(recon_losses)
        self.recon_std = np.std(recon_losses)
        np.save(
            file=os.path.join(self.save_dir, f'{self.model_name}_recon_threshold.npy'),
            arr=self.recon_threshold_value,
        )
        np.save(
            file=os.path.join(self.save_dir, f'{self.model_name}_recon_mean.npy'),
            arr=self.recon_mean,
        )
        np.save(
            file=os.path.join(self.save_dir, f'{self.model_name}_recon_std.npy'),
            arr=self.recon_std,
        )
        return None

    def _get_root_cause_threshold_encoder(self, xs: Tensor) -> None:
        self.eval()
        us_list = []
        
        with torch.no_grad():
            for x in xs:
                us = self._testing_step(x)[-1]
                us_list.append(us.cpu().numpy())
        
        us_all = np.concatenate(us_list, axis=0).reshape(-1, self.num_vars)
        self.lower_encoder = np.quantile(a=us_all, q=(1 - self.root_cause_threshold_encoder) / 2, axis=0)
        self.upper_encoder = np.quantile(a=us_all, q=1 - (1 - self.root_cause_threshold_encoder) / 2, axis=0)
        self.us_mean_encoder = np.median(us_all, axis=0)
        self.us_std_encoder = np.std(us_all, axis=0)
        np.save(
            file=os.path.join(self.save_dir, f'{self.model_name}_lower_encoder.npy'),
            arr=self.lower_encoder,
        )
        np.save(
            file=os.path.join(self.save_dir, f'{self.model_name}_upper_encoder.npy'),
            arr=self.upper_encoder,
        )
        np.save(
            file=os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy'),
            arr=self.us_mean_encoder,
        )
        np.save(
            file=os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy'),
            arr=self.us_std_encoder,
        )

        return None

    def _get_root_cause_threshold_decoder(self, xs: Tensor) -> None:
        self.eval()
        diff_list = []
        
        with torch.no_grad():
            for x in xs:
                _, nexts_hat, nexts, _, _, _, _, _ = self._testing_step(x=x, add_u=False)
                diff = (nexts - nexts_hat).cpu().numpy().ravel()
                diff_list.append(diff)
            
        us_all = np.concatenate(diff_list, axis=0).reshape(-1, self.num_vars)
        self.lower_decoder = np.quantile(a=us_all, q=(1 - self.root_cause_threshold_decoder) / 2, axis=0)
        self.upper_decoder = np.quantile(a=us_all, q=1 - (1 - self.root_cause_threshold_decoder) / 2, axis=0)
        self.us_mean_decoder = np.mean(us_all, axis=0)
        self.us_std_decoder = np.std(us_all, axis=0)
        
        np.save(
            file=os.path.join(self.save_dir, f'{self.model_name}_lower_decoder.npy'),
            arr=self.lower_decoder,
        )
        np.save(
            file=os.path.join(self.save_dir, f'{self.model_name}_upper_decoder.npy'),
            arr=self.upper_decoder,
        )
        np.save(
            file=os.path.join(self.save_dir, f'{self.model_name}_us_mean_decoder.npy'),
            arr=self.us_mean_decoder,
        )
        np.save(
            file=os.path.join(self.save_dir, f'{self.model_name}_us_std_decoder.npy'),
            arr=self.us_std_decoder,
        )
        
        return None
    
    def _testing_root_cause(
            self,
            xs: Tensor,
            labels: Tensor,
    )-> None:
        self.load_state_dict(
            state_dict=torch.load(
                f=os.path.join(self.save_dir, f'{self.model_name}.pt'),
                map_location=self.device,
            )
        )
        self.eval()
        self.us_mean_encoder = np.load(
            file=os.path.join(self.save_dir, f'{self.model_name}_us_mean_encoder.npy')
        )
        self.us_std_encoder = np.load(
            file=os.path.join(self.save_dir, f'{self.model_name}_us_std_encoder.npy')
        )

        us_list = []
        us_sample_list = []

        with torch.no_grad():
            for i in range(len(xs)):
                x = xs[i]
                label = labels[i]
                us = self._testing_step(x=x, label=label, add_u=False)[-1]
                us_sample_list.append(us[self.window_size:].cpu().numpy())
                us_list.append(us.cpu().numpy())
        
        us_all = np.concatenate(us_list, axis=0).reshape(-1, self.num_vars)
        self._log_and_print('=' * 50)
        us_all_z_score = (-(us_all - self.us_mean_encoder) / self.us_std_encoder)
        us_all_z_score_pot = []

        for i in range(self.num_vars):
            pot_val, _ = pot(
                data=us_all_z_score[:, i],
                risk=self.risk,
                init_level=self.initial_leval,
                num_candidates=self.num_candidates,
            )
            us_all_z_score_pot.append(pot_val)
        us_all_z_score_pot = np.array(us_all_z_score_pot)

        k_all = []
        k_at_step_all = []

        for i in range(len(xs)):
            us_sample = us_sample_list[i]
            z_scores = (-(us_sample - self.us_mean_encoder) / self.us_std_encoder)
            k_lst = topk(
                z_scores=z_scores,
                label=labels[i][self.window_size * 2:],
                threshold=us_all_z_score_pot,
            )
            k_at_step = topk_at_step(
                scores=z_scores,
                labels=labels[i][self.window_size * 2:],
            )
            k_all.append(k_lst)
            k_at_step_all.append(k_at_step)
        
        k_all = np.array(k_all).mean(axis=0)
        k_at_step_all = np.array(k_at_step_all).mean(axis=0)
        ac_at = [k_at_step_all[0], k_at_step_all[2], k_at_step_all[4], k_at_step_all[9]]
        self._log_and_print(f'Root cause analysis AC@1: {ac_at[0]:.5f}')
        self._log_and_print(f'Root cause analysis AC@3: {ac_at[1]:.5f}')
        self._log_and_print(f'Root cause analysis AC@5: {ac_at[2]:.5f}')
        self._log_and_print(f'Root cause analysis AC@10: {ac_at[3]:.5f}')
        self._log_and_print(f'Root cause analysis Avg@10: {np.mean(k_at_step_all):.5f}')

        ac_star_at = [k_all[0], k_all[9], k_all[99], k_all[499]]
        self._log_and_print(f'Root cause analysis AC*@1: {ac_star_at[0]:.5f}')
        self._log_and_print(f'Root cause analysis AC*@10: {ac_star_at[1]:.5f}')
        self._log_and_print(f'Root cause analysis AC*@100: {ac_star_at[2]:.5f}')
        self._log_and_print(f'Root cause analysis AC*@500: {ac_star_at[3]:.5f}')
        self._log_and_print(f'Root cause analysis Avg*@500: {np.mean(k_all):.5f}')

        return None
    
    def _testing_causal_discover(
        self,
        xs: Tensor,
        causal_struct_value: Array
    ) -> None:
        self.load_state_dict(
            state_dict=torch.load(
                f=os.path.join(self.save_dir, f'{self.model_name}.pt'),
                map_location=self.device,
            )
        )
        self.eval()
        encoder_causal_list = []

        with torch.no_grad():
            for x in xs:
                _, _, _, encoder_coeffs, _, _, _, _ = self._testing_step(x)
                encoder_estimate = torch.max(
                    input=torch.median(torch.abs(encoder_coeffs), dim=0)[0],
                    dim=0
                ).values.cpu().numpy()
                encoder_causal_list.append(encoder_estimate)
        
        encoder_causal_struct_estimate_lst = np.stack(
            arrays=encoder_causal_list,
            axis=0,
        )

        encoder_auroc = []
        encoder_auprc = []
        encoder_hamming = []
        encoder_f1 = []

        for i in range(len(encoder_causal_struct_estimate_lst)):
            encoder_auroc_temp, encoder_auprc_temp = eval_causal_structure(
                a_true=causal_struct_value,
                a_pred=encoder_causal_struct_estimate_lst[i],
            )
            encoder_auroc.append(encoder_auroc_temp)
            encoder_auprc.append(encoder_auprc_temp)
            encoder_q = np.quantile(
                a=encoder_causal_struct_estimate_lst[i],
                q=self.causal_quantile,
            )
            encoder_a_hat_binary = (encoder_causal_struct_estimate_lst[i] >= encoder_q).astype(float)
            _, _, _, _, ham_e = eval_causal_structure_binary(
                a_true=causal_struct_value,
                a_pred=encoder_a_hat_binary,
            )
            encoder_hamming.append(ham_e)
            encoder_f1.append(f1_score(causal_struct_value.flatten(), encoder_a_hat_binary.flatten()))

        self._log_and_print(f'Causal discovery F1: {np.mean(encoder_f1):.5f} with std: {np.std(encoder_f1):.5f}')
        self._log_and_print(f'Causal discovery AUROC: {np.mean(encoder_auroc):.5f} with std: {np.std(encoder_auroc):.5f}')
        self._log_and_print(f'Causal discovery AUPRC: {np.mean(encoder_auprc):.5f} with std: {np.std(encoder_auprc):.5f}')
        self._log_and_print(f'Causal discovery Hamming Distance: {np.mean(encoder_hamming):.5f} with std: {np.std(encoder_hamming):.5f}')
        
        return None

import attr
import math
import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd 
import collections

import copy
import logging

logger = logging.getLogger("tensor2struct")

# Todo: move this class for universal use
@attr.s
class SpiderEncoderState:
    state = attr.ib()
    memory = attr.ib()
    question_memory = attr.ib()
    schema_memory = attr.ib()
    words_for_copying = attr.ib()

    pointer_memories = attr.ib()
    pointer_maps = attr.ib()

    m2c_align_mat = attr.ib()
    m2t_align_mat = attr.ib()

    # for copying
    tokenizer = attr.ib()

    def find_word_occurrences(self, token):
        occurrences = [i for i, w in enumerate(self.words_for_copying) if w == token]
        if len(occurrences) > 0:
            return occurrences[0]
        else:
            return None

class DeepEnsembleModelAgnostic(nn.Module):
    def __init__(
        self,  
        device=None, 
        num_particles=2,
    ):
        super().__init__()
        self.inner_steps = 1
        self.num_particles = num_particles
        self.device = device

    def ensemble_train(self, model, 
                   model_encoder_params,
                   model_aligner_params,
                   model_decoder_params, 
                   batch,
                   num_batch_accumulated):
        
        return self.particles_base_train(model, 
                               model_encoder_params, 
                               model_aligner_params,
                               model_decoder_params, 
                               batch,
                               num_batch_accumulated)

    def particles_base_train(self, 
                   model, 
                   model_encoder_params,
                   model_aligner_params,
                   model_decoder_params,
                   batch,
                   num_batch_accumulated):
        assert model.training
        
        encoder_params = []
        for i in range(self.num_particles):
            encoder_params.append(list(model.list_of_encoders[i].parameters()))
        params_matrix = torch.stack(
            [torch.nn.utils.parameters_to_vector(params) for params in encoder_params],
            dim=0
        )
        aligner_params = list(model.aligner.parameters())
        decoder_params = list(model.decoder.parameters())
        particle_len = len(encoder_params[0])
        aligner_len = len(aligner_params)
        aligner_p_vec = torch.nn.utils.parameters_to_vector(aligner_params)
        decoder_p_vec = torch.nn.utils.parameters_to_vector(decoder_params)
        ret_dic = {}
        for _step in range(self.inner_steps):
            # for computing distance 
            distance_nll = torch.empty(size=(self.num_particles,
                                             params_matrix.size(1)),
                                       device=self.device)
            # decoder grad vector, store decoder grads
            alinger_grads_vec = torch.zeros_like(aligner_p_vec)
            decoder_grads_vec = torch.zeros_like(decoder_p_vec)
            enc_input_list = [enc_input for enc_input, dec_output in batch]
            column_pointer_maps = [
                {i: [i] for i in range(len(desc["columns"]))} for desc in enc_input_list
            ]
            table_pointer_maps = [
                {i: [i] for i in range(len(desc["tables"]))} for desc in enc_input_list
            ]

            final_losses = []
            for i in range(self.num_particles):
                
                enc_states = []
                # for single input source domain
                plm_output = model.bert_model(enc_input_list)[0]
                enc_input, dec_output = batch[0]
                relation = model.schema_linking(enc_input)
                relation = model.schema_linking(enc_input)
                (
                    q_enc_new_item,
                    c_enc_new_item,
                    t_enc_new_item,
                ) = model.list_of_encoders[i](enc_input, 
                                                    plm_output,
                                                    relation)
                # attention memory 
                memory = []
                include_in_memory = model.list_of_encoders[0].include_in_memory
                if "question" in include_in_memory:
                    memory.append(q_enc_new_item)
                if "column" in include_in_memory:
                    memory.append(c_enc_new_item)
                if "table" in include_in_memory:
                    memory.append(t_enc_new_item)
                memory = torch.cat(memory, dim=1)
                # alignment matrix
                align_mat_item = model.aligner(
                    enc_input, q_enc_new_item, c_enc_new_item, t_enc_new_item, relation
                )
        
                enc_states.append(
                    SpiderEncoderState(
                        state=None,
                        words_for_copying=enc_input["question_for_copying"],
                        tokenizer=model.list_of_encoders[0].tokenizer,
                        memory=memory,
                        question_memory=q_enc_new_item,
                        schema_memory=torch.cat((c_enc_new_item, t_enc_new_item), dim=1),
                        pointer_memories={
                            "column": c_enc_new_item,
                            "table": t_enc_new_item,
                        },
                        pointer_maps={
                            "column": column_pointer_maps[0],
                            "table": table_pointer_maps[0],
                        },
                        m2c_align_mat=align_mat_item[0],
                        m2t_align_mat=align_mat_item[1],
                    )
                )

                losses = []
                for enc_state, (enc_input, dec_output) in zip(enc_states, batch):
                    ret_dic = model.decoder(dec_output, enc_state)
                    losses.append(ret_dic["loss"])
                loss = torch.mean(torch.stack(losses, dim=0), dim=0) / num_batch_accumulated
                final_losses.append(loss.item())
                enc_dec_grads = torch.autograd.grad(loss, 
                                                    encoder_params[i] + aligner_params + decoder_params,
                                                    allow_unused=True)
                particle_grads = enc_dec_grads[:particle_len]
                aligner_grads = list(enc_dec_grads[particle_len:particle_len + aligner_len])
                for idx, g in enumerate(aligner_grads):
                    if g is None:
                        aligner_grads[idx] = torch.zeros_like(aligner_params[idx])
                decoder_grads = list(enc_dec_grads[particle_len + aligner_len:])
                for idx, g in enumerate(decoder_grads):
                    if g is None:
                        decoder_grads[idx] = torch.zeros_like(decoder_params[idx])
                alinger_grads_vec = alinger_grads_vec + (1/self.num_particles)*torch.nn.utils.parameters_to_vector(aligner_grads)
                decoder_grads_vec = decoder_grads_vec + (1/self.num_particles)*torch.nn.utils.parameters_to_vector(decoder_grads)
                
                distance_nll[i, :] = torch.nn.utils.parameters_to_vector(particle_grads)
            
            kernel_matrix, grad_kernel, _ = DeepEnsembleModelAgnostic.get_kernel(params=params_matrix,
                                              num_of_particles=self.num_particles)
            
            # compute inner gradients with rbf kernel
            encoders_grads = torch.matmul(kernel_matrix, distance_nll) - grad_kernel
            # copy inner_grads to main network
            for i in range(self.num_particles):
                for p_tar, p_src in zip(model_encoder_params[i],
                                        DeepEnsembleModelAgnostic.vector_to_list_params(encoders_grads[i],
                                                                                             model_encoder_params[i])):
                    p_tar.grad.data.add_(p_src) # todo: divide by num_of_sample if inner is in ba
            # copy aligner grads to the main network
            for p_tar, p_src in zip(model_aligner_params,
                            DeepEnsembleModelAgnostic.vector_to_list_params(alinger_grads_vec, model_aligner_params)):
                p_tar.grad.data.add_(p_src)
            # copy decoder grads to the main network
            for p_tar, p_src in zip(model_decoder_params,
                            DeepEnsembleModelAgnostic.vector_to_list_params(decoder_grads_vec, model_decoder_params)):
                p_tar.grad.data.add_(p_src)
            # trying to free gpu memory 
            # not sure it would help
            # del kernel_matrix
            # del grad_kernel
            # del distance_nll
            # del inner_grads
            # torch.cuda.empty_cache()
            
        logger.info(f"Inner loss: {sum(final_losses)/self.num_particles}")
        ret_dic["loss"] = sum(final_losses)/self.num_particles
        
        return ret_dic
    
    @staticmethod
    def get_kernel(params: torch.Tensor, num_of_particles):
        """
        Compute the RBF kernel for the input
        
        Args:
            params: a tensor of shape (N, M)
        
        Returns: kernel_matrix = tensor of shape (N, N)
        """
        pairwise_d_matrix = DeepEnsembleModelAgnostic.get_pairwise_distance_matrix(x=params)

        median_dist = torch.quantile(input=pairwise_d_matrix, q=0.5)  # tf.reduce_mean(euclidean_dists) ** 2
        h = median_dist / np.log(num_of_particles)

        kernel_matrix = torch.exp(-pairwise_d_matrix / h)
        kernel_sum = torch.sum(input=kernel_matrix, dim=1, keepdim=True)
        grad_kernel = -torch.matmul(kernel_matrix, params)
        grad_kernel += params * kernel_sum
        grad_kernel /= h

        return kernel_matrix, grad_kernel, h

    @staticmethod
    def get_pairwise_distance_matrix(x: torch.Tensor) -> torch.Tensor:
        """Calculate the pairwise distance between each row of tensor x
        
        Args:
            x: input tensor
        
        Return: matrix of point-wise distances
        """
        n, m = x.shape

        # initialize matrix of pairwise distances as a N x N matrix
        pairwise_d_matrix = torch.zeros(size=(n, n), device=x.device)

        # num_particles = particle_tensor.shape[0]
        euclidean_dists = torch.nn.functional.pdist(input=x, p=2) # shape of (N)

        # assign upper-triangle part
        triu_indices = torch.triu_indices(row=n, col=n, offset=1)
        pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

        # assign lower-triangle part
        pairwise_d_matrix = torch.transpose(pairwise_d_matrix, dim0=0, dim1=1)
        pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

        return pairwise_d_matrix
    
    @staticmethod
    def vector_to_list_params(vector, other_params):
    
        params = []
        
        # pointer for each layer params
        pointer = 0 
        
        for param in other_params:
            # total number of params each layer
            num_params = int(np.prod(param.shape))
            
            params.append(vector[pointer:pointer+num_params].view(param.shape))
            
            pointer += num_params

        return params
    
DEMA = DeepEnsembleModelAgnostic
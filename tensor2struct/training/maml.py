import math
import torch
import torch.nn as nn
import collections

import copy
import higher
import logging

logger = logging.getLogger("tensor2struct")


class ModelAgnosticMetaLearning(nn.Module):
    def __init__(
        self, model=None, inner_opt=None, first_order=False, device=None,
    ):
        super().__init__()
        self.inner_opt = inner_opt
        self.first_order = first_order
        self.inner_steps = 1
        self.device = device

    def get_inner_opt_params(self):
        """
        Equvalent to self.parameters()
        """
        return []

    def meta_train(self, model, inner_batch, outer_batches):
        assert not self.first_order
        return self.maml_train(model, inner_batch, outer_batches)

    def maml_train(self, model, inner_batch, outer_batches):
        assert model.training
        ret_dic = {}
        with higher.innerloop_ctx(
            model, self.inner_opt, copy_initial_weights=False, device=self.device
        ) as (fmodel, diffopt), torch.backends.cudnn.flags(enabled=False):
            for _step in range(self.inner_steps):
                inner_ret_dic = fmodel(inner_batch)
                inner_loss = inner_ret_dic["loss"]

                # use the snippet for checking higher
                # def test(params):
                #     params = [p for p in params if p.requires_grad]
                #     all_grads = torch.autograd.grad(
                #         loss,
                #         params,
                #         retain_graph=True,
                #         allow_unused=True,
                #     )
                #     print(len(params), sum(p is not None for p in all_grads))
                # import pdb; pdb.set_trace()
                # test(model.parameters())
                # test(fmodel.fast_params)

                diffopt.step(inner_loss)
            logger.info(f"Inner loss: {inner_loss.item()}")

            mean_outer_loss = torch.Tensor([0.0]).to(self.device)
            with torch.set_grad_enabled(model.training):
                for batch_id, outer_batch in enumerate(outer_batches):
                    outer_ret_dic = fmodel(outer_batch)
                    mean_outer_loss += outer_ret_dic["loss"]
            mean_outer_loss.div_(len(outer_batches))
            logger.info(f"Outer loss: {mean_outer_loss.item()}")

            final_loss = inner_loss + mean_outer_loss
            final_loss.backward()

            # not sure if it helps
            del fmodel
            import gc

            gc.collect()

        ret_dic["loss"] = final_loss.item()
        return ret_dic


MAML = ModelAgnosticMetaLearning

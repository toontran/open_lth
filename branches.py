from general_setups import *

class RandomPruneBranch(Branch):
    def branch_function(self, seed: int, strategy: str = 'layerwise', start_at: str = 'rewind',
                        layers_to_ignore: str = ''):
        # Randomize the mask.
        mask = Mask.load(self.level_root)

        # Randomize while keeping the same layerwise proportions as the original mask.
        if strategy == 'layerwise': mask = Mask(shuffle_state_dict(mask, seed=seed))

        # Randomize globally throughout all prunable layers.
        elif strategy == 'global': mask = Mask(unvectorize(shuffle_tensor(vectorize(mask), seed=seed), mask))

        # Randomize evenly across all layers.
        elif strategy == 'even':
            sparsity = mask.sparsity
            for i, k in sorted(mask.keys()):
                layer_mask = torch.where(torch.arange(mask[k].size) < torch.ceil(sparsity * mask[k].size),
                                         torch.ones_like(mask[k].size), torch.zeros_like(mask[k].size))
                mask[k] = shuffle_tensor(layer_mask, seed=seed+i).reshape(mask[k].size)

        elif strategy == "2dfilterwise":
            for i, key in enumerate(sorted(mask.keys())):
                if "conv" in key:
                    mask[key] = torch.cat([torch.cat([shuffle_tensor(filter_2d, seed=seed+k+len(filter_3d)*j+len(filter_3d)*len(mask[key])*i).unsqueeze(0)
                                                    for k, filter_2d in enumerate(filter_3d)]).unsqueeze(0)
                                        for j, filter_3d in enumerate(mask[key])])

        elif strategy == "3dfilterwise":
            for i, key in enumerate(sorted(mask.keys())):
                if "conv" in key:
                    mask[key] = torch.cat([
                        shuffle_tensor(filter_3d, seed=seed+j+len(mask[key])*i).unsqueeze(0) 
                                                      for j, filter_3d in enumerate(mask[key])
                    ])

        elif strategy == "3dfilterpos":
            keys = [k for k in mask.keys() if "conv" in k]
            for i in range(len(keys)-1):
                perm = torch.randperm(mask[keys[i]].size()[0])
                mask[keys[i]] = mask[keys[i]][perm]
                mask[keys[i+1]] = mask[keys[i+1]][:,perm]

        elif strategy == "3dfilterpos & 2dfilterwise":
            keys = [k for k in mask.keys() if "conv" in k]
            for i in range(len(keys)-1):
                perm = torch.randperm(mask[keys[i]].size()[0])
                mask[keys[i]] = mask[keys[i]][perm]
                mask[keys[i+1]] = mask[keys[i+1]][:,perm]

            for i, key in enumerate(sorted(mask.keys())):
                if "conv" in key:
                    mask[key] = torch.cat([torch.cat([shuffle_tensor(filter_2d, seed=seed+k+len(filter_3d)*j+len(filter_3d)*len(mask[key])*i).unsqueeze(0)
                                                    for k, filter_2d in enumerate(filter_3d)]).unsqueeze(0)
                                        for j, filter_3d in enumerate(mask[key])])


        # Identity.
        elif strategy == 'identity': pass

        # Error.
        else: raise ValueError(f'Invalid strategy: {strategy}')

        # Reset the masks of any layers that shouldn't be pruned.
        if layers_to_ignore:
            for k in layers_to_ignore.split(','): mask[k] = torch.ones_like(mask[k])

        # Save the new mask.
        mask.save(self.branch_root)

        # Determine the start step.
        if start_at == 'init':
            start_step = self.lottery_desc.str_to_step('0ep')
            state_step = start_step
        elif start_at == 'end':
            start_step = self.lottery_desc.str_to_step('0ep')
            state_step = self.lottery_desc.train_end_step
        elif start_at == 'rewind':
            start_step = self.lottery_desc.train_start_step
            state_step = start_step
        else:
            raise ValueError(f'Invalid starting point {start_at}')

        # Train the model with the new mask.
        model = PrunedModel(model_registry_load(self.level_root, state_step, self.lottery_desc.model_hparams), mask)
        standard_train(model, self.branch_root, self.lottery_desc.dataset_hparams,
                             self.lottery_desc.training_hparams, start_step=start_step, verbose=self.verbose)

    @staticmethod
    def description():
        return "Randomly prune the model."

    @staticmethod
    def name():
        return 'randomly_prune'

class RandomInitBranch(Branch):
    def branch_function(self, start_at_step_zero: bool = False):
        model = PrunedModel(model_registry_get(self.lottery_desc.model_hparams),
                            Mask.load(self.level_root))
        start_step = self.lottery_desc.str_to_step('0it') if start_at_step_zero else self.lottery_desc.train_start_step
        Mask.load(self.level_root).save(self.branch_root)
        standard_train(model, self.branch_root, self.lottery_desc.dataset_hparams,
                             self.lottery_desc.training_hparams,
                       start_step=start_step, verbose=self.verbose)

    @staticmethod
    def description():
        return "Randomly reinitialize the model."

    @staticmethod
    def name():
        return 'randomly_reinitialize'

class ExternalBranch(Branch):
    def branch_function(self, mask_path: str, model_path: str, start_step:str="0it"):
        # Need to load masks somehow
        model = model_registry_get(self.lottery_desc.model_hparams)
        def device_str():
            # GPU device.
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                device_ids = ','.join([str(x) for x in range(torch.cuda.device_count())])
                return f'cuda:{device_ids}'
            # CPU device.
            else:
                return 'cpu'
        torch_device = torch.device(device_str())
        state_dict = torch.load(model_path, map_location=torch_device)
        # import pdb;pdb.set_trace()
        # Remove "_mask" in parameter names
        try:
            model.load_state_dict({name.replace("_mask", "") : state_dict[name] for name in state_dict})
        except Exception as e:
            print(e)
            model.load_state_dict({name.replace("_mask", "") : state_dict[name] for name in state_dict}, strict=False)
        mask = Mask(torch.load(mask_path, map_location=torch_device))
        model.to(torch_device)
        #import pdb;pdb.set_trace()
        pruned_model = PrunedModel(model, mask)
        #mask.save(self.branch_root)
        standard_train(pruned_model, self.branch_root, self.lottery_desc.dataset_hparams,
                       self.lottery_desc.training_hparams,
                       start_step=self.lottery_desc.str_to_step(start_step), 
                       verbose=self.verbose)

    @staticmethod
    def description():
        return "Load external models and masks for evaluation."

    @staticmethod
    def name():
        return 'external'
















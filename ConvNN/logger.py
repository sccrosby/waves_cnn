"""
WandBLogger is a wrapper around Weights and Biases functions for ease of use.
"""

import wandb

class WandBLogger:
    def __init__(self, project, name, tags=None, notes=None, entity=None,
                 config=None):
        """
        Params:
            project: name of the wandb project
            name: name of the experiment
            tags: tags to add to the experiment (string or list of strings)
            notes: a note to add to the experiment (string)
            config: initial config for the experiment, not required
            entity: 'entity' is a group on wandb, if you aren't part of a group
                    you probably don't need to use this
        """

        self.run = wandb.init(
                project=project,
                name=name,
                tags=tags,
                notes=notes,
                entity=entity,
                config=config,
            )

    def log_params(self, params):
        """
        This function updates the experiment's parameters
        params should be a dictionary
        """
        self.run.config.update(params)

    def log_metrics(self, metrics, step=None, commit=True):
        """
        This function logs metrics from the train or validation loop.
        Params:
            metrics: a dictionary comtaining metrics to log
            step: the name of the variable which is used as the default x axis
            commit: if commit is False, you can call this function multiple
                    times on the same step
        """
        self.run.log(
                metrics,
                step=None if step is None else metrics[step],
                commit=commit,
            )

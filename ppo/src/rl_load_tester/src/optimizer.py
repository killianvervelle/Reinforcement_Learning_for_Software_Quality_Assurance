import optuna


class Optimizer:
    def __init__(self, env, agent, model):
        self.env = env
        self.agent = agent
        self.model = model

    def objective(self, trial):
        params = {
            "discount_factor": trial.suggest_float('discount_factor', 0.8, 1.0),
            "learning_rate": trial.suggest_loguniform('learning_rate', 1e-4, 1e-3),
            "batch_size": trial.suggest_categorical('batch_size', [32, 64, 128]),
            "epsilon": trial.suggest_float('epsilon', 0.2, 0.4),
            "entropy_coefficient": trial.suggest_float('entropy_coefficient', 0.01, 0.1),
            "hidden_dimensions": trial.suggest_categorical('hidden_dimensions', [32, 64, 128]),
            "dropout": trial.suggest_float('dropout', 0.2, 0.5),
            "max_grad_norm": trial.suggest_categorical('max_grad_norm', [1.0, 5.0, 10.0, 50.0]),
        }

        env_train, env_test = self.env.initialize_env(self.model)

        n_test_rewards = self.agent.run_agent(
            env_train=env_train,
            env_test=env_test,
            discount_factor=params["discount_factor"],
            epsilon=params["epsilon"],
            entropy_coefficient=params["entropy_coefficient"],
            hidden_dimensions=params["hidden_dimensions"],
            dropout=params["dropout"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            max_grad_norm=params["max_grad_norm"]
        )

        return n_test_rewards

    def optimize_hyperparameters(self):
        study = optuna.create_study(direction='maximize')

        study.optimize(self.objective, n_trials=10)

        print("Best hyperparameters: ", study.best_params)
        print("Best test reward: ", study.best_value)

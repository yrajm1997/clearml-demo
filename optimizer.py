from clearml import Task
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange, DiscreteParameterRange
from clearml.automation import GridSearch, RandomSearch, HyperParameterOptimizer

task = Task.init(
    project_name='MNIST Hyper-Parameter Optimization',
    task_name='Automatic Hyper-Parameter Optimization',
    task_type=Task.TaskTypes.optimizer,
)

optimizer = HyperParameterOptimizer(
      # specifying the task to be optimized, task must be in system already so it can be cloned
      base_task_id="9f8967c52c11467ab4dbc5974ac17493", 
      # setting the hyperparameters to optimize
      hyper_parameters=[
          DiscreteParameterRange(name='General/optimizer', values=['adam', 'sgd']),
          UniformParameterRange(name='General/dropout_rate', min_value=0.01, max_value=0.5, step_size=0.05),
          UniformIntegerParameterRange(name='General/hidden_layers', min_value=1, max_value=3, step_size=1),
          UniformIntegerParameterRange(name='General/neurons', min_value=70, max_value=150, step_size=10),
          UniformIntegerParameterRange(name='General/epochs', min_value=3, max_value=5, step_size=1),          
          ],
      # setting the objective metric we want to maximize/minimize
      objective_metric_title='epoch_accuracy',
      objective_metric_series='validation: epoch_accuracy',
      objective_metric_sign='max',

      # setting optimizer
      optimizer_class=RandomSearch,           # SearchStrategy optimizer to use for the hyperparameter search
  
      # configuring optimization parameters
      max_number_of_concurrent_tasks=2,  
      optimization_time_limit=10.,            # The maximum time (in minutes) for the entire optimization process
      execution_queue='default',              # Execution queue to use for launching Tasks
      )


# This will automatically create and print the optimizer new task id for later use.
# If a Task was already created, it will use it.
optimizer.start()

# wait until optimization completes or time-out
optimizer.wait()

# make sure we stop all jobs
optimizer.stop()

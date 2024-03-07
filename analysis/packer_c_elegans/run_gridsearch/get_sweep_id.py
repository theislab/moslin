import wandb
import yaml
import click


@click.command()
@click.argument("config_yaml")
@click.argument("train_file")
@click.argument("project_name")
def get_id(config_yaml, train_file, project_name):

    # read config from yaml file
    with open(config_yaml) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config['program'] = train_file

    # initialize this sweep on wandb
    sweep_id = wandb.sweep(sweep=config, project=project_name)

    return sweep_id


if __name__ == '__main__':
    get_id()

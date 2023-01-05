import argparse


def main(params):
    pass

def get_args() -> dict:
    parser = argparse.ArgumentParser(description='Rubiks Cube RL')

    # rubiks cube parameters
    parser.add_argument("--num_sides", type=int, help="Number of sides of the Rubiks Cube", default=3)

    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    return training_config


if __name__ == "__main__":
    main(get_args())
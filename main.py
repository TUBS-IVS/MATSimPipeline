import argparse
from pipelines.popsim_to_raw_plans import popsim_to_raw_plans_main
from utils import matsim_pipeline_setup
#from transaction_pipeline import transaction_main
from utils.logger import logging
logger = logging.getLogger(__name__)

# Define the available pipelines in a dictionary for easy reference and scalability
PIPELINES = {
    "user_data": popsim_to_raw_plans_main,
    #"transaction": transaction_main
}


def main(args):
    selected_pipelines = PIPELINES.keys() if args.all else args.pipelines
    matsim_pipeline_setup.create_output_directory()
    for pipeline_name in selected_pipelines:
        try:
            if pipeline_name in PIPELINES:
                logger.info(f"Starting {pipeline_name} pipeline...")
                PIPELINES[pipeline_name](args)
                logger.info(f"Completed {pipeline_name} pipeline.")
            else:
                logger.warning(f"Unknown pipeline: {pipeline_name}")
        except Exception as e:
            logger.error(f"Error running {pipeline_name} pipeline: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specific data pipelines.")

    # Dropdown list of pipelines to choose from
    parser.add_argument("--pipelines", nargs='+', choices=PIPELINES.keys(),
                        help="Specify which pipelines to run. Can list multiple separated by space.")

    # Option to run all pipelines
    parser.add_argument("--all", action="store_true", help="Run all available pipelines.")

    # Any other arguments for the pipelines can be added here
    parser.add_argument("--some_arg", type=str, help="Some argument that might be used by a pipeline.")

    args = parser.parse_args()

    if not args.all and not args.pipelines:
        logger.error("You must specify either specific pipelines to run or use the --all option.")
    else:
        main(args)




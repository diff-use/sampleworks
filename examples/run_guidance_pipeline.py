import argparse

from sampleworks.utils.guidance_script_utils import run_guidance_job_queue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-queue-path", type=str, required=True,
        help="Path to the job queue pickle file, which should contain a list of "
             "GuidanceConfig objects to run"
    )
    args = parser.parse_args()
    run_guidance_job_queue(args.job_queue_path)

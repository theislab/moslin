#!/bin/bash

# Initialize variables with default values
config_yaml=""
train_file=""
project_name="Unnamed"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config_yaml)
      config_yaml="$2"
      shift 2
      ;;
    -t|--train_file)
      train_file="$2"
      shift 2
      ;;
    -p|--project_name)
      project_name="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check for required arguments
if [ -z "$config_yaml" ] || [ -z "$train_file" ]; then
  echo "Usage: $0 -c|--config_yaml <path_to_config_yaml> -t|--train_file <path_to_train_file> [-p|--project_name <project_name>]"
  exit 1
fi

# Print the argument values
echo "Using the following arguments:"
echo "config_yaml: $config_yaml"
echo "train_file: $train_file"
echo "project_name: $project_name"

# Run the Python script and capture its output
# output=$(python -c "from get_sweep_id_2 import get_id; print(get_id())")
output=$(python get_sweep_id.py "$config_yaml" "$train_file" "$project_name")

# Extract the last line from the output using 'tail'
sweep_url=$(echo "$output" | tail -n 1)

# Extract the unique identifier from the output using 'grep'
sweep_id=$(echo "$sweep_url" | grep -oP '(?<=/sweeps/)[^"]+')
# sweep_id=$output

# Print the last line
echo "Retrieved the following sweep_id: $sweep_id"

# call the submission script
sbatch request_resources.sh "$sweep_id" "$project_name"

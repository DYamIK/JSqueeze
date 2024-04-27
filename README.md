# JSqueeze

## Requirements
At least `python>=3.6, <3.7` is required.
``` bash
pip install -r requirements.txt
```

## Datasets

Datasets `B1, B2, B3` are on [Simulation](https://zenodo.org/record/8153367) (updated on 2023-07-17).
Datasets `B5, B6, B7, B8` are available at [Real](https://zenodo.org/record/8153849)
The ground truth root cause sets are in `injection_info.csv` in each subfolder.

## Usage

```
$python run_algorithm.py --help
Usage: run_algorithm.py [OPTIONS]

  :param name: :param input_path: :param output_path: :param num_workers:
  :param kwargs: :return:

Options:
  --name TEXT            name of this setting
  --input-path TEXT      will read data from {input_path}/{name}
  --output-path TEXT     if {output_path} is a dir, save to
                         {output_path}/{name}.json; otherwise save to
                         {output_path}
  --num-workers INTEGER  num of processes
  --derived              means we should read {timestamp}.a.csv and
                         {timestamp}.b.csv
  --help                 Show this message and exit.
```

``` 
$python run_evaluation.py --help
Usage: run_evaluation.py [OPTIONS]

Options:
  -i, --injection-info TEXT  injection_info.csv file
  -p, --predict TEXT         output json file
  -c, --config TEXT          config json file
  -o, --output-path TEXT     output path
  --help                     Show this message and exit.
```

The config json file should contain the attribute names, e.g.:

```
{
  "columns": [
    "a", "b", "c", "d"
  ]
}
```

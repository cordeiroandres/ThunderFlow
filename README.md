# EV-battery-estimation-calculator

This code estimates electric vehicle (EV) battery consumption and speed based solely on GPS coordinates and timestamps. It processes the data to calculate distances and time intervals between points, allowing for the computation of average speeds. An energy consumption model is applied, considering speed and battery characteristics, to estimate energy usage. The code provides real-time consumption estimates and displays them to the user, making it a valuable tool for monitoring and optimizing EV performance during trips.

This tool has different options to do the calculation, it has two important methods, you can upload a dataset with coordinates and time intervals or put just a simply trajectory to do the estimation.

## Installation

You can install this package using pip:
[![pip install ThunderFlowPro](https://img.shields.io/badge/pip%20install-ThunderFlowPro-brightgreen)](https://pypi.org/project/ThunderFlowPro/)

```bash
pip install ThunderFlowPro
```

## Usage
```python
import ThunderFlowPro as T

# Example usage

lst_traj=T.consumption(df,
                CreateTrajectories=True,                               
                temporal_thr=1200,
                spatial_thr=50,
                minpoints=4,
                MapMatching='valhalla',
                ResultsByTrajectory=True
                )
```


# For example
![Tutorial](https://github.com/cordeiroandres/EV-battery-calculator/blob/main/Images/Tutorial1.png)

![Tutorial](https://github.com/cordeiroandres/EV-battery-calculator/blob/main/Images/Tutorial2.png)

![Tutorial](https://github.com/cordeiroandres/EV-battery-calculator/blob/main/Images/Tutorial3.png)

![Tutorial](https://github.com/cordeiroandres/EV-battery-calculator/blob/main/Images/Tutorial4.png)

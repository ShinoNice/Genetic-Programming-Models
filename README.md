# Inference of Lactose Contents Using Genetic Programming

Welcome to the Neural and Evolutionary Learning project! This project was developed as part of the NEL course for the 2023/2024 academic year. Our objective is to use Genetic Programming models to compare and discuss the behaviour, performance, and application of different GPs.

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)

## Project Overview
The primary objective of the project is to compare and discuss the behaviour, performance, and application of different GP models. For this purpose, the dataset located in the datamart folder should be utilised. 
This dataset has data on cow milk production from a farm in North Italy, which is entirely based on Automatic Milking Systems (AMSs). In the AMS adopted in this farm, cows decide when they will be milked. During each milking event, the milking robot acquires extensive data on the cow level, productivity and milking behaviour. In particular, milk contents (fat, protein, lactose) are measured at each milking event and are used to evaluate the milk quality. The measurement of these components by the milking robot is not straightforward: it is done with a colourimetric method, and the robot needs to be calibrated every 2 weeks with chemical measurements from the laboratory.

The modelling problem to be solved in the project is to infer the lactose contents exclusively using data from the milking robots. The features dataset is provided in the data_project_nel.csv file. Lactose target values are provided in the y_lactose.csv file.

## Data
The data used for this project is provided is the following:
- The features dataset is provided in the data_project_nel.csv file.
- Lactose target values are provided in the y_lactose.csv file.
- Additionally, you can also work with models for Fat (y_fat.csv) and Protein (y_protein.csv) contents.

## Methodology
1. **Data Preprocessing**: Cleaning and preparing the data for analysis.
2. **Model Training**: Training genetic programming models like GP, GSGP and NEAT.
3. **Evaluation**: Assessing the model's performance using appropriate metrics.

## Results
The final model's performance and key findings are detailed in the `NEL_Report.pdf`.

For any questions or further information, feel free to contact me via GitHub.

Happy coding!

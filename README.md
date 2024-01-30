# MARS-2024
Code Repository for an Experimental Project in the MARS Program

## (Tentative) Project Timeline

1. Create model training pipeline.
2. Recreate backdoored models via various techniques.
3. Test SLT claims on backdoored models.

This is very liable to change.

## Directory Structure

/data: Store third-party data sets.  

/models: Save trained models (both clean and backdoored) in .pt or .onnx formats.  

/notebooks: Jupyter notebooks for exploratory data analysis, prototyping, and demonstrations.  

/src:  
    /src/models: Python modules for defining model architectures.  
    /src/data: Scripts for data downloading, preprocessing, and loading.  
    /src/training: Code for model training, including clean and backdoored models.  
    /src/backdoor: Code specific to embedding backdoors in models.  
    /src/evaluation: Scripts for model evaluation, including learning coefficient estimation.  
    /src/utils: Utility functions used across the project.  
    /src/vizualization: Visualization scripts for results and data.  
/tests: Unit tests and integration tests for your code.  

## Notes

- Had to clone devinterp package and 'pip install .' to get it to work with newer versions of torch.
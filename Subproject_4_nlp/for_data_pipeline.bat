@echo off
pushd %~dp0
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo --------------------------------------------
echo Step 1: Extracting data from parquet...
python main.py || (
    echo main.py failed. Exiting.
    exit /b
)

echo --------------------------------------------
echo Step 1 completed. Checking output...
if exist output_data\data_for_read.csv (
    echo Output file found: output_data\data_for_read.csv
) else (
    echo File not found. Please check main.py.
    pause
    exit /b
)

pause

echo --------------------------------------------
echo Step 2: Running basic preprocessing...
python preprocessing_for_models\preprocessing_model1.py || (
    echo Preprocessing failed.
    pause
    exit /b
)

echo --------------------------------------------
echo Preprocessing done. CSVs should be generated.
pause
popd
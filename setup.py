pip install virtualenv
virtualenv python3 env
python -r install requirements.txt
python -m ipykernel install --user --name env --display-name "ENV"

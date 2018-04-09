# DSI-Capstone
Capstone work submission for General Assembly Data Science Intensive program


To run the project follow these instructions:

1. clone this repo 

2. `cd DSI-Capstone/`

3. create a virtual environment with pyhton 3 named 'env' (this part is very important):

create: `virtualenv -p python3 env`
activate: `source env/bin/activate`

4. install requirements: 

`pip install -r requirements.txt`

5. now to make jupyter notebook use our environment install it as a kernel:

`python -m ipykernel install --user --name env --display-name "ENV"`

6. open bookmates.ipynb with Jupyter notebook and then select ENV from the kernel settings: kernel -> change kernel -> ENV


7. run notebook! note: it will take couple minutes for it to finish as we are going to load chromedrive and perform quite a few crawling jobs.  


This repo has been tested on Linux and if those steps are followed correcly everything should work. I cannot vouch for Mac, and god forbid you try setting it up on Windows! 




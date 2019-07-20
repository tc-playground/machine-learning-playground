# Install NLPIA

### Install Python3

* http://docs.python-guide.org/en/latest/starting/install3/osx/

  ```
  homebrew [install|upgrade] python 
  ```

### Install PipEnv

* http://docs.python-guide.org/en/latest/dev/virtualenvs/#virtualenvironments-ref

  ```
  pip3 install --user pipenv
  export PATH=$PATH:$(python3 -m site --user-base)/bin
  ...
  pipenv install requests
  pipenv install -r requirements-all.txt
  pipenv install -r requirements.txt --skip-lock
  pipenv shell
  ```


### Install VirtualEnv

* http://docs.python-guide.org/en/latest/dev/virtualenvs/#virtualenvironments-ref

  ```
  pip3 install virtualenv
  virtualenv --version
  ...
  virtualenv -p /usr/local/bin/python3 venv
  source venv/bin/activate
  deactivate
  ```

### Install Anaconda3

* https://docs.anaconda.com/anaconda/install/mac-os

  ```
  curl https://repo.continuum.io/archive/Anaconda3-5.1.0-MacOSX-x86_64.sh -o Anaconda3-5.1.0-MacOSX-x86_64.sh
  chmod +x Anaconda3-5.1.0-MacOSX-x86_64.sh
  ./Anaconda3-5.1.0-MacOSX-x86_64.sh 
  ```

### Instal NLPIA

* https://github.com/totalgood/nlpia
 
  ```
  brew install portaudio cmu-pocketsphinx swig
  git clone https://github.com/totalgood/nlpia.git && cd nlpia
  conda env create -f conda/environment.yml
  source activate conda_env_nlpia
  ```
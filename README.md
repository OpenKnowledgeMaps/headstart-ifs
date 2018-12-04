## Setup

### Install docker and docker-compose

* https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce-1
```
sudo apt-get remove docker docker-engine docker.io
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce
```
* https://docs.docker.com/compose/install/
```
sudo curl -L "https://github.com/docker/compose/releases/download/1.23.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Install Cassandra

* http://cassandra.apache.org/download/
```
echo "deb http://www.apache.org/dist/cassandra/debian 311x main" | sudo tee -a /etc/apt/sources.list.d/cassandra.sources.list
curl https://www.apache.org/dist/cassandra/KEYS | sudo apt-key add -
sudo apt-get update
sudo apt-get install cassandra
```
### Optional: install kubernetes and minikube



### Download/Git clone the repo




## Usage

### Start cassandra

`sudo service cassandra start`

### Docker-compose

`sudo service docker-compose build`

`sudo service docker-compose up -d`

DO THIS ONLY IN THE BEGINNING! DATA WILL BE LOST
`docker-compose exec ifs python manage.py recreate_db`

`sudo service docker-compose down`

### Alternative: kubernetes

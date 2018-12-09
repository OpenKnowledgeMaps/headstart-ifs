## Setup

### Install docker and docker-compose

* Note: some of the commands may have to be run with `sudo`
* https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce-1
```
apt-get remove docker docker-engine docker.io
apt-get update
apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
apt-get update
apt-get install docker-ce
```
* https://docs.docker.com/compose/install/
```
curl -L "https://github.com/docker/compose/releases/download/1.23.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
```

### Install Cassandra

* http://cassandra.apache.org/download/
```
echo "deb http://www.apache.org/dist/cassandra/debian 311x main" | tee -a /etc/apt/sources.list.d/cassandra.sources.list
curl https://www.apache.org/dist/cassandra/KEYS | apt-key add -
apt-get update
apt-get install cassandra
```
### Optional: install kubernetes and minikube



### Download/Git clone the repo




## Usage

### Start cassandra

`service cassandra start`

### Docker-compose

* build images
```
docker-compose build
```

* start services and send them to th docker daemon
```
docker-compose up -d
```

* create pristine database: DO THIS ONLY IN THE BEGINNING! DATA WILL BE LOST
```
docker-compose exec ifs python manage.py recreate_db
```

* shut service down
```
docker-compose down
```

### Alternative: kubernetes

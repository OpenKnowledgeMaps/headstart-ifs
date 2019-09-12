FROM rocker/r-ver:3.5.1

MAINTAINER Chris Kittel "christopher.kittel@openknowledgemaps.org"
RUN apt-get update
RUN apt-get -y install libssl-dev
RUN apt-get -y install libxml2-dev

RUN R -e 'install.packages("remotes", repos = c(CRAN = "https://cran.rstudio.com"))'
RUN R -e 'remotes::install_github("rstudio/renv@0.6.0-98")'

WORKDIR /headstart
COPY renv.lock ./
RUN R -e 'options(repos="https://cran.wu.ac.at"); renv::restore()'
COPY other-scripts ./backend

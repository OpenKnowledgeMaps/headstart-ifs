FROM rocker/shiny:3.5.1

MAINTAINER Chris Kittel "christopher.kittel@openknowledgemaps.org"

RUN R -e 'install.packages("remotes", repos = c(CRAN = "https://cran.rstudio.com"))'
RUN R -e 'remotes::install_github("rstudio/renv")'

WORKDIR /headstart
COPY renv.lock ./
RUN R -e 'renv::restore()'
COPY other-scripts ./backend

# Packages ####

library(ggplot2)
library(dplyr)
library(gglander)

# Data ####

data(credit_data, package = 'modeldata')
credit <- credit_data |> tibble::as_tibble()
credit
?modeldata::credit_data

# EDA ####

credit |> count(Marital)
credit |> count(Home)
credit |> count(Job)
credit |> count(Status)

ggplot(credit, aes(x=Status, fill=Status)) + geom_bar()

ggplot(credit, aes(x=Status, y=Amount)) + geom_violin()

ggplot(credit, aes(x=Income, y=Amount)) + geom_point()
ggplot(credit, aes(x=log(Income), y=log(Amount))) + geom_point()
ggplot(credit, aes(x=log(Income), y=log(Amount))) + geom_point() + geom_smooth()
ggplot(credit, aes(x=log(Income), y=log(Amount), color=Job)) + geom_point() + geom_smooth()
ggplot(credit, aes(x=log(Income), y=log(Amount), color=Job)) + 
    geom_point(show.legend = FALSE) + 
    geom_smooth(show.legend = FALSE) + 
    facet_wrap(~Job)

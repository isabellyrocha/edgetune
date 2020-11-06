all <- read.csv("~/Documents/edgetune/workloads/results/all.out", header=FALSE)
names(all) <- c("device", "network", "database", "duration", "energy")
View(all)

cifar10 <- all[all$database == "cifar10", ]
cifar100 <- all[all$database == "cifar100", ]

ggplot(all, aes(x=device, y=energy, fill = database)) + 
  geom_bar(stat="identity", color="black", 
           position=position_dodge()) + 
  theme_bw() + theme(axis.title.x = element_blank(), legend.position="top") + ylab("Duration [s]") +
  facet_wrap(~ network, ncol = 5, scales = "free_y")
ggsave("~/Documents/edgetune/exps/plots/duration.pdf")

ggplot(all, aes(x=device, y=energy, fill = database)) + 
  geom_bar(stat="identity", color="black", 
           position=position_dodge()) + 
  theme_bw() + theme(axis.title.x = element_blank(), legend.position="top") + ylab("Energy [J]") +
  facet_wrap(~ network, ncol = 5, scales = "free_y")
ggsave("~/Documents/edgetune/exps/plots/energy.pdf")


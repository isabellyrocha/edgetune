exp <- read.csv("~/Documents/edgetune/results/exp_64.csv", header=FALSE)
names(exp) <- c("cores", "mem", "layers", "batch", "server", "edge")
View(exp)

png(file="~/Documents/edgetune/results/exp_64.png",
    width=600, height=350)
plot(exp$server, pch=19, col="red", type="b", xlab="", ylab="Inferente time [s]", main="Max samples = 64")
lines(exp$edge, pch=18, col="blue", type="b", lty=2)
legend(1, 3, legend=c("Server", "Edge"),
       col=c("red", "blue"), lty=1:2, cex=0.8)
dev.off()

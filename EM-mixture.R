#Andrieu, C., de Freitas, N., Doucet, A. et al. An Introduction to MCMC for Machine Learning. Machine Learning 50, 5–43 (2003). https://doi.org/10.1023/A:1020281327116
library(ggplot2)
library(dplyr)

#True 모델 생성
n_cluster <- 5
param_df <- data.frame(mu=rnorm(n_cluster, sd=5), sigma2=rchisq(n_cluster, 5), pi=rmultinom(1, 100, rep(1, n_cluster)/n_cluster)/100)

#Density를 그리기 위한 작업
plot_x_range <- seq(min(param_df$mu)-10, max(param_df$mu)+10, length.out=100)
plot(plot_x_range,rowSums(apply(param_df, 1, function(par) {dnorm(plot_x_range, par[1], sqrt(par[2]))*par[3]})), type = 'l', main = "GMM pdf", ylab = '', xlab='x')

#표본 생성
n_sample <- 100
gmm_ind <- sample(1:n_cluster, n_sample, prob = param_df$pi, replace = T)
gmm_sample <- sapply(gmm_ind, function(ind) {rnorm(1, param_df[ind,1], sqrt(param_df[ind,2]))})

plot(plot_x_range,rowSums(apply(param_df, 1, function(par) {dnorm(plot_x_range, par[1], sqrt(par[2]))*par[3]})), type = 'l', main = "GMM pdf", ylab = '', xlab='x')
hist(gmm_sample, probability = T, add = T, col=rgb(0,0,0, alpha=0.3), breaks = 20)

#분산에 대한 MLE.
var_mle <- function(x) {var(x) * (length(x) - 1) / length(x)}

#초기 값
param_est <- data.frame(mu=rnorm(n_cluster, sd=5), sigma2=rchisq(n_cluster, 5), pi=rmultinom(1, 100, rep(1, n_cluster)/n_cluster)/100)
n_mcmc <- 10

for (it in 1:100) {
  #E step
  #잠재 변수를 샘플링하기 위한 각 클러스터에 들어갈 확률
  alphas <- apply(param_df, 1, function(par) {dnorm(gmm_sample, par[1], sqrt(par[2]))*par[3]})
  alphas <- alphas/rowSums(alphas)
  #잠재변수 추출(어느 분포/클러스터에서 나왔는지)
  sample_est_ind <- apply(alphas, 1, function(ap) {sample(1:n_cluster, n_mcmc, prob = ap, replace = T)})
  
  #M step
  #잠재변수가 가리키는 클러스터에 대응하는 데이터를 각 클러스터에 반영
  param_est_next <- data.frame(y=rep(gmm_sample, each=n_mcmc), cls=as.vector(sample_est_ind)) %>%
    group_by(cls) %>% summarise(mu=mean(y), sigma2=var_mle(y), pi=n()) %>% as.data.frame() %>% select(-cls)
  param_est <- param_est_next
  param_est[,3] <- param_est[,3]/sum(param_est[,3])
}

param_df
param_est

#샘플 히스토그램+True density+estimate density
ggplot()+
  geom_line(aes(x=plot_x_range,y=rowSums(apply(param_df, 1, function(par) {dnorm(plot_x_range, par[1], sqrt(par[2]))*par[3]}))))+
  geom_line(aes(x=plot_x_range,y=rowSums(apply(param_est, 1, function(par) {dnorm(plot_x_range, par[1], sqrt(par[2]))*par[3]}))), color='red')+
  geom_histogram(aes(x=gmm_sample, y=after_stat(density)), alpha=0.3)+ylab("density")+xlab("X")


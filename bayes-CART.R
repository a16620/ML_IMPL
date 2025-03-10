library(data.tree)
library(rpart)
library(dplyr)
library(ggplot2)
library(ggpubr)

#순서
#1.MCMC 동작
#2.MCMC를 위한 동작 정의 (!!먼저 실행!!)

##################### regression tree #####################

library(ISLR)
data("Carseats")
str(Carseats)

train.idx <- sample(1:nrow(Carseats), as.integer(0.2*nrow(Carseats)))
train.data <- Carseats[train.idx,]
test.data <- Carseats[-train.idx,]

#Parameters for prior
mc.params <- CART.extract.params(Sales~., train.data)

#Record tree performance
criteria.name <- c("l.post", "SSE", "n.leaf")
criteria.func <- function(tree) {
  c(CART.prob.likelihood.regression(tree, mc.params)+CART.prob.prior(tree), CART.get.tree.SSE(tree), tree$leafCount)
}

#Initialize MCMC
mtree <- CART.create(train.data, c(0.5, 2.5))
SSE0 <- CART.get.tree.SSE(mtree)
print(SSE0)
print(CART.prob.likelihood.regression(mtree, mc.params))

#MCMC
max.iter <- 1000
criteria.matrix <- `colnames<-`(matrix(0, ncol=length(criteria.name), nrow=max.iter+1), criteria.name)
criteria.matrix[1,] <- criteria.func(mtree)

optimal.SSE <- criteria.matrix[1,2]
optimal.model <- mtree

a.count <- 0
vec.moves <- c("grow", "prune", "change", "swap")
fail.count <- setNames(rep(0, length(vec.moves)), vec.moves)
for (iter in 1:max.iter) {
  mc.move <- sample(vec.moves, 1)
  if (mc.move == "grow") {
    mc.new <- CART.move.grow(mtree)
  } else if (mc.move == "prune") {
    mc.new <- CART.move.prune(mtree)
  } else if (mc.move == "change") {
    mc.new <- CART.move.change(mtree)
  } else if (mc.move == "swap") {
    mc.new <- CART.move.swap(mtree)
  }
  
  if (is.null(mc.new) || !CART.check.tree.ok(mc.new$tree.new)) {
    fail.count[mc.move] <- fail.count[mc.move]+1
    criteria.matrix[iter+1,] <- criteria.matrix[iter,]
    next
  }
  
  l.a.ratio.u <- (mc.new$l.prob.rev+CART.prob.likelihood.regression(mc.new$tree.new, mc.params)+CART.prob.prior(mc.new$tree.new))
  l.a.ratio.l <- (mc.new$l.prob+CART.prob.likelihood.regression(mtree, mc.params)+CART.prob.prior(mtree))
  if (log(runif(1)) <= (l.a.ratio.u-l.a.ratio.l)) {
    mtree <- mc.new$tree.new
    a.count <- a.count + 1
  }
  
  criteria.matrix[iter+1,] <- criteria.func(mtree)
  
  if (optimal.SSE > criteria.matrix[iter+1,2] || (optimal.SSE == criteria.matrix[iter+1,2] && runif(1) <= 0.5)) {
    optimal.SSE <- criteria.matrix[iter+1,2]
    optima.model <- Clone(mtree)
  }
  
  if (iter %% 100 == 0) {
    print(paste0('============= #', as.character(iter), '(', format(a.count/iter, digits = 3),')'), quot=F)
    print(CART.prob.likelihood.regression(mtree, mc.params), quot=F)
    print(SSE0-mean(criteria.matrix[(iter-99):(iter+1),2]), quot=F)
  }
}

criteria.df <- as.data.frame(criteria.matrix)
ggarrange(ggplot(criteria.df) + geom_line(aes(x=0:max.iter, y=l.post))+xlab('iter')+theme_classic(),
          ggplot(criteria.df) + geom_line(aes(x=0:max.iter, y=SSE))+xlab('iter')+theme_classic(),
          ggplot(criteria.df) + geom_line(aes(x=0:max.iter, y=n.leaf))+xlab('iter')+theme_classic(),
          ncol = 1)


greed.tree <- rpart(Sales~., data=train.data, method="anova",
                    control = rpart.control(maxdepth = max(Get(optima.model$leaves, "level"))))
greed.pred <- predict(greed.tree, newdata = test.data)

CART.set.predict(optima.model)
bayes.pred <- CART.get.predict(optima.model, test.data)

print(mean((greed.pred-test.data[,1])**2))
print(mean((bayes.pred-test.data[,1])**2))

##################### classification tree #####################

#Data loading
cancer <- read.csv('./data.csv')
cancer$diagnosis <- as.factor(cancer$diagnosis)
cancer <- cancer[,-c(1,33)]

str(cancer)

train.idx <- sample(1:nrow(cancer), as.integer(0.2*nrow(cancer)))
train.data <- cancer[train.idx,]
test.data <- cancer[-train.idx,]

#Parameters for prior
mc.params <- CART.extract.params(diagnosis~., train.data, is.categorical = T)

#Record tree performance
criteria.name <- c("l.post", "miscl", "n.leaf")
criteria.func <- function(tree) {
  c(CART.prob.likelihood.category(tree, mc.params)+CART.prob.prior(tree), CART.get.tree.miscl(tree), tree$leafCount)
}

#Initialize MCMC
mtree <- CART.create(train.data, c(0.5, 5))
print(CART.prob.likelihood.category(mtree, mc.params))

#MCMC
max.iter <- 1000
criteria.matrix <- `colnames<-`(matrix(0, ncol=length(criteria.name), nrow=max.iter+1), criteria.name)
criteria.matrix[1,] <- criteria.func(mtree)

optimal.miscl <- criteria.matrix[1,2]
optimal.model <- mtree

a.count <- 0
vec.moves <- c("grow", "prune", "change", "swap")
fail.count <- setNames(rep(0, length(vec.moves)), vec.moves)
for (iter in 1:max.iter) {
  mc.move <- sample(vec.moves, 1)
  if (mc.move == "grow") {
    mc.new <- CART.move.grow(mtree)
  } else if (mc.move == "prune") {
    mc.new <- CART.move.prune(mtree)
  } else if (mc.move == "change") {
    mc.new <- CART.move.change(mtree)
  } else if (mc.move == "swap") {
    mc.new <- CART.move.swap(mtree)
  }
  
  if (is.null(mc.new) || !CART.check.tree.ok(mc.new$tree.new)) {
    criteria.matrix[iter+1,] <- criteria.matrix[iter,]
    fail.count[mc.move] <- fail.count[mc.move]+1
    next
  }
  
  l.a.ratio.u <- (mc.new$l.prob.rev+CART.prob.likelihood.category(mc.new$tree.new, mc.params)+CART.prob.prior(mc.new$tree.new))
  l.a.ratio.l <- (mc.new$l.prob+CART.prob.likelihood.category(mtree, mc.params)+CART.prob.prior(mtree))
  if (log(runif(1)) <= (l.a.ratio.u-l.a.ratio.l)) {
    mtree <- mc.new$tree.new
    a.count <- a.count + 1
  }
  
  criteria.matrix[iter+1,] <- criteria.func(mtree)
  
  if (optimal.miscl > criteria.matrix[iter+1,2] || (optimal.miscl == criteria.matrix[iter+1,2] && runif(1) <= 0.5)) {
    optimal.miscl <- criteria.matrix[iter+1,2]
    optima.model <- Clone(mtree)
  }
  
  if (iter %% 100 == 0) {
    print(paste0('============= #', as.character(iter), '(', format(a.count/iter, digits = 3),')'), quot=F)
    print(CART.prob.likelihood.category(mtree, mc.params), quot=F)
    print(CART.get.tree.miscl(mtree), quot=F)
  }
}

criteria.df <- as.data.frame(criteria.matrix)
ggarrange(ggplot(criteria.df) + geom_line(aes(x=0:max.iter, y=l.post))+xlab('iter')+theme_classic(),
          ggplot(criteria.df) + geom_line(aes(x=0:max.iter, y=miscl))+xlab('iter')+theme_classic(),
          ggplot(criteria.df) + geom_line(aes(x=0:max.iter, y=n.leaf))+xlab('iter')+theme_classic(),
          ncol = 1)

greed.tree <- rpart(diagnosis~., data=train.data, method="class",
                    control = rpart.control(maxdepth = max(Get(optima.model$leaves, "level"))))
greed.pred <- predict(greed.tree, newdata = test.data, type = "class")

CART.set.predict(optima.model)
bayes.pred <- CART.get.predict(optima.model, test.data)

print(mean(greed.pred==test.data[,1]))
print(mean(bayes.pred==test.data[,1]))

##################### Probability #####################

CART.prob.likelihood.regression <- function(tree, params, verb=F) {
  c.a <- params$a
  c.b <- tree$leafCount
  c.c <- ifelse(is.null(params$c), 0, as.numeric(params$c))
  c.mu <- params$mu
  c.nu <- params$nu
  c.lambda <- params$lambda
  
  l.prob <- c.c+0.5*c.b*log(c.a)-0.5*sum(sapply(tree$leaves, function(node) log(length(node$obs.idx))))-
    0.5*(nrow(tree$full.obs)+c.nu)*log(sum(sapply(tree$leaves, function(node) {
      node.obs <- CART.get.obs(node)
      n.i <- nrow(node.obs)
      y.i <- node.obs[,1]
      s.i <- CART.get.node.SSE(node)
      t.i <- (mean(y.i)-c.mu)**2*c.a*n.i/(c.a+n.i)
      return(t.i+s.i)
    }))+c.nu*c.lambda)
  return(l.prob)
}

#V2
CART.prob.likelihood.regression <- function(tree, params) {
  c.a <- params$a
  c.c <- ifelse(is.null(params$c), 0, as.numeric(params$c))
  c.mu <- params$mu
  c.nu <- params$nu
  c.nuMlamb <- params$lambda*c.nu
  c.aMnu <- 0.5*log(c.a)-lgamma(c.nu/2)
  
  l.prob <- c.c+0.5*c.nu*log(c.nuMlamb)+
    sum(sapply(tree$leaves, function(node) {
      n.i <- length(node$obs.idx)
      c.aMnu-0.5*log(n.i+c.a)+lgamma((n.i+c.nu)/2)
      }))-
    0.5*sum(sapply(tree$leaves, function(node) {
      node.obs <- CART.get.obs(node)
      n.i <- nrow(node.obs)
      y.i <- node.obs[,1]
      s.i <- CART.get.node.SSE(node)
      t.i <- (mean(y.i)-c.mu)**2*c.a*n.i/(c.a+n.i)
      return((n.i+c.nu)*log(t.i+s.i+c.nuMlamb))
    }))
  return(l.prob)
}

CART.prob.likelihood.category <- function(tree, params, verb=F) {
  c.a <- params$a
  c.as <- params$a.s
  c.ap <- params$a.p
  l.prob <- sum(sapply(tree$leaves, function(node) {
      node.obs <- CART.get.obs(node)
      n.i <- as.numeric(table(node.obs[,1]))
      if (verb) {
        print(n.i)
      }
      return(c.ap+sum(lgamma(n.i+c.a))-lgamma(sum(n.i)+c.as))
    }))
  return(l.prob)
}

CART.prob.prior <- function(tree) {
  split.param <- tree$param.split
  
  sum(unlist(tree$Get(function(node) {
    p <- split.param[1]*node$level**(-split.param[2])
    log(ifelse(isLeaf(node), 1-p,p))
  })))
}

CART.prob.select.rule <- function(obs, len.rule.values, is.categorical) {
  if (len.rule.values <= 1 || is.na(len.rule.values)) {
    return(0)
  }
  if (is.categorical) {
    return(-log(2**len.rule.values-2)-log(ncol(obs)-1))
  } else {
    return(-log(len.rule.values)-log(ncol(obs)-1))
  }
}

##################### MCMC functions #####################

CART.set.predict <- function(tree) {
  if (is.factor(tree$full.obs[,1])) {
    pv <- lapply(tree$leaves, function(node) {
      tb.val <- table(CART.get.obs(node)[,1])
      node$pred.val <- names(which.max(tb.val))
    })
  } else {
    pv <- lapply(tree$leaves, function(node) {
      node$pred.val <- mean(CART.get.obs(node)[,1])
    })
  }
}

CART.get.predict <- function(tree, data) {
  pred.val <- numeric(nrow(data))
  for (i in 1:nrow(data)) {
    row <- data[i,]
    node <- tree
    while(isNotLeaf(node)) {
      if (node$split.rule$fun(row)) {
        node <- node$children[[1]]
      } else {
        node <- node$children[[2]]
      }
    }
    pred.val[i] <- node$pred.val
  }
  return(pred.val)
}

CART.check.tree.ok <- function(tree) {
  all(sapply(tree$leaves, function(node) {length(node$obs.idx) > 0}))
}

CART.get.tree.SSE <- function(tree) {
  sum(sapply(tree$leaves, CART.get.node.SSE))
}

CART.get.node.SSE <- function(node) {
  obs.idx <- node$obs.idx
  if (length(obs.idx) == 0) {
    return(0)
  } else {
    y <- CART.get.obs(node)[,1]
    if (length(y) > 1) {
      return(var(y)*(length(y)-1))
    }
    return(0)
  }
}

CART.get.tree.miscl <- function(tree) {
  n.total <- nrow(tree$full.obs)
  sum(sapply(tree$leaves, function(node) {
    CART.get.node.miscl(node)*length(node$obs.idx)/n.total
  }))
}

CART.get.node.miscl <- function(node) {
  obs.idx <- node$obs.idx
  if (length(obs.idx) == 0) {
    return(0)
  } else {
    y <- CART.get.obs(node)[,1]
    p <- as.numeric(table(y))
    p <- p/sum(p)
    return(1-max(p))
  }
  
}

CART.extract.params <- function(formula., data, is.categorical=F) {
  if (is.categorical) {
    Y <- data[, all.vars(formula.)[1]]
    if (T) {
      a <- rep(1, length(levels(Y)))
    } else {
      a <- as.numeric(table(Y))
    }
    
    return(list(
      a=a,
      a.s=sum(a),
      a.p=lgamma(sum(a))-sum(lgamma(a)),
      y.cont=F
    ))
  } else {
    fit.tree = rpart(formula., data=data, method="anova", cp=0.008)
    var.est <- var(fit.tree$y)
    var.min <- data.frame(leaf=fit.tree$where, y=fit.tree$y) %>% group_by(leaf) %>% summarise(var=var(y)) %>% pull(var) %>% min()
    return(list(
      mu=mean(fit.tree$y),
      a=var.min/var.est,
      lambda=var.min,
      nu=length(unique(fit.tree$where)),
      y.cont=T
    ))
  }
}

CART.rule2name <- function(obs, split.col, split.value) {
  obs.filt.col <- obs[,split.col]
  if (is.factor(obs.filt.col)) {
    lvs <- levels(obs.filt.col)
    lvs.key <- lvs %in% split.value
    subl <- lvs[lvs.key]
    subr <- lvs[!lvs.key]
    
    return(c(
      paste0(split.col, "={", paste(subl,collapse=","), '}'),
      paste0(split.col, "={", paste(subr,collapse=","), '}')
    ))
  } else {
    return(c(
      paste0(split.col, "<=", as.character(split.value)),
      paste0(split.col, ">", as.character(split.value))
    ))
  }
}


CART.select.rule <- function(node) {
  obs <- CART.get.obs(node)
  col.candidates <- colnames(obs)[-1]
  while (length(col.candidates) > 0) {
    rule.colname <- sample(col.candidates, 1)
    if (is.factor(obs[,rule.colname])) {
      rule.values <- unique(obs[,rule.colname])
      len.values <- length(rule.values)
      if (len.values < 2) {
        col.candidates <- col.candidates[col.candidates != rule.colname]
        next
      }
      rule.value <- rule.values[sample(1:len.values, sample(1:(len.values-1), 1))]
      
      return(
        list(
          split.col=rule.colname,
          split.value=sort(rule.value)
        )
      )
    } else {
      #rule.values <- sort(obs[,rule.colname], decreasing = T)[-1] #최대값은 빈 노드를 만들 수 있음.
      #len.rule.values <- length(rule.values)
      #if (len.rule.values <= 1 || rule.values[1] == rule.values[len.rule.values]) {
      #  col.candidates <- col.candidates[col.candidates != rule.colname]
      #  next
      #}
      #rule.value <- sample(rule.values, 1)
      
      rule.values <- sort(obs[,rule.colname])
      len.rule.values <- length(rule.values)
      
      if (len.rule.values <= 2 || rule.values[1] == rule.values[len.rule.values]) {
        col.candidates <- col.candidates[col.candidates != rule.colname]
        next
      }
      
      val.idx <- sample(1:(length(rule.values)-1), 1)
      val.aux <- runif(1)
      rule.value <- rule.values[val.idx]+val.aux*(rule.values[val.idx+1]-rule.values[val.idx])
      if (rule.value == rule.values[val.idx+1]) {
        rule.value <- (rule.values[val.idx]+rule.values[val.idx+1])/2
      }
      return(
        list(
          split.col=rule.colname,
          split.value=rule.value
        )
      )
    }
  }
  return(NULL)
}

CART.update.obs <- function(tree) {
  tree$Do(function(node) {
    obs.key <- node$split.rule$fun(CART.get.obs(node))

    CART.set.obs(node$children[[1]], node$obs.idx[obs.key])
    CART.set.obs(node$children[[2]], node$obs.idx[!obs.key])
  }, traversal="level", filterFun = isNotLeaf)
  return(CART.check.tree.ok(tree))
}

CART.get.prob.rule <- function(node) {
  obs <- CART.get.obs(node)
  rule.colname <- node$split.rule$rule.col
  if (is.factor(obs[,rule.colname])) {
    len.rule.values <- length(unique(obs[,rule.colname]))
    return(CART.prob.select.rule(obs, len.rule.values, T))
  } else {
    return(CART.prob.select.rule(obs, nrow(obs)-1, F))
  }
}

CART.update.rule.swap <- function(node, rule.info) {
  node$split.rule <- rule.info
  
  child.names <- CART.rule2name(CART.get.obs(node), rule.info$split.col, rule.info$split.value)
  names(node$children) <- child.names
  node$children[[1]]$name <- child.names[1]
  node$children[[2]]$name <- child.names[2]
}

CART.update.rule <- function(node, rule.info) {
  rule <- function(obs) {
    obs.filt.col <- obs[,rule.info$split.col]
    if (is.factor(obs.filt.col)) {
      return(obs.filt.col %in% rule.info$split.value)
    }
    return(obs.filt.col <= rule.info$split.value)
  }
  
  obs <- CART.get.obs(node)
  obs.key <- rule(obs)
  if (all(obs.key) || !any(obs.key))
    return(F)
  
  rule.info$fun <- rule
  node$split.rule <- rule.info
  
  child.names <- CART.rule2name(obs, rule.info$split.col, rule.info$split.value)
  names(node$children) <- child.names
  node$children[[1]]$name <- child.names[1]
  node$children[[2]]$name <- child.names[2]
  return(T)
}

CART.set.rule <- function(node, rule.info) {
  rule <- function(obs) {
    obs.filt.col <- obs[,rule.info$split.col]
    if (is.factor(obs.filt.col)) {
      return(obs.filt.col %in% rule.info$split.value)
    }
    return(obs.filt.col <= rule.info$split.value)
  }
  
  obs <- CART.get.obs(node)
  obs.key <- rule(obs)
  if (all(obs.key) || !any(obs.key))
    return(F)
  
  rule.info$fun <- rule
  node$split.rule <- rule.info
  left.obs.idx <- node$obs.idx[obs.key]
  right.obs.idx <- node$obs.idx[!obs.key]
  
  child.names <- CART.rule2name(obs, rule.info$split.col, rule.info$split.value)
  CART.set.obs(node$AddChild(child.names[1]), left.obs.idx)
  CART.set.obs(node$AddChild(child.names[2]), right.obs.idx)
  
  return(T)
}

CART.get.obs <- function(node) {
  return(node$root$full.obs[node$obs.idx,])
}

CART.set.obs <- function(node, obs.idx) {
  node$obs.idx <- obs.idx
  return(node)
}

CART.create <- function(obs, param.split=c(0.95, 0.5)) {
  root <- CART.set.obs(Node$new("Root", param.split=param.split, full.obs=obs), 1:nrow(obs))
  return(root)
}

as.probability <- function(unp) {
  unp/sum(unp)
}

CART.compare.rule <- function(rule1, rule2) {
  if (rule1$split.col != rule2$split.col)
    return(F)
  if (length(rule1$split.value) != length(rule2$split.value))
    return(F)
  if (any(rule1$split.value != rule1$split.value))
    return(F)
  
  return(T)
}

##################### MCMC moves #####################

CART.move.grow <- function(tree) {
  tree.new <- Clone(tree)
  
  terminal.nodes <- tree.new$leaves
  node.level <- sapply(terminal.nodes, function(node) node$level)
  
  split.param <- tree.new$param.split
  node.prob <- as.probability(node.level**(-split.param[2]))
  
  while (length(terminal.nodes) > 0) {
    selected.node.idx <- sample(1:length(node.prob), 1, prob = node.prob)
    selected.node <- terminal.nodes[[selected.node.idx]]
    rule <- CART.select.rule(selected.node)
    if (is.null(rule)) {
      terminal.nodes <- terminal.nodes[-selected.node.idx]
      node.prob <- as.probability(node.prob[-selected.node.idx])
      next
    }
    if (CART.set.rule(selected.node, rule)) {
      break
    }
  }
  
  if (length(terminal.nodes) == 0) {
    return(NULL)
  }
  
  node.prob.rev <- as.probability(1/node.prob)
  return(list(
    tree.new=tree.new,
    l.prob=log(node.prob[selected.node.idx])+CART.get.prob.rule(selected.node),
    l.prob.rev=log(node.prob.rev[selected.node.idx])
  ))
}

CART.move.prune <- function(tree) {
  if (tree$leafCount == 1) {
    return(NULL)
  }
  tree.new <- Clone(tree)
  
  parent.of <- Traverse(tree.new, filterFun = function(node) {
    isNotLeaf(node) && isLeaf(node$children[[1]]) && isLeaf(node$children[[2]])
  })
  node.level <- sapply(parent.of, function(node) node$level)
  node.prob <- as.probability(node.level**tree.new$param.split[2])
  
  selected.node.idx <- sample(1:length(node.prob), 1, prob = node.prob)
  selected.node <- parent.of[[selected.node.idx]]
  
  if (Prune(selected.node, function(cn) F) != 2)
    print("Warning: Pruned not 2")
  l.prob.sel.split <- CART.get.prob.rule(selected.node)
  selected.node$split.rule <- NULL
  
  node.prob.rev <- as.probability(1/node.prob)
  
  
  return(list(
    tree.new=tree.new,
    l.prob=log(node.prob[selected.node.idx]),
    l.prob.rev=log(node.prob.rev[selected.node.idx])+l.prob.sel.split
  ))
}

CART.move.change <- function(tree) {
  if (tree$leafCount == 1) {
    return(NULL)
  }
  tree.new <- Clone(tree)
  split.node <- Traverse(tree.new, filterFun = isNotLeaf)
  node.level <- sapply(split.node, function(node) node$level)
  node.prob <- as.probability(node.level**tree.new$param.split[2])
  
  if (length(split.node) == 0)
    return(NULL)
  ok <- F
  for (max.try in 1:1000) {
    selected.node.idx <- sample(1:length(node.prob), 1)#, prob = node.prob)
    selected.node <- split.node[[selected.node.idx]]
    rule <- CART.select.rule(selected.node)
    if (is.null(rule))
      next
    
    subtree <- Clone(selected.node)
    subtree$full.obs <- tree.new$full.obs
    
    if (!CART.update.rule(subtree, rule) || !CART.update.obs(subtree)) {
      next
    }
    ok <- T
    if (isNotRoot(selected.node)) {
      subtree$full.obs <- NULL
      parent.of.no <- selected.node$parent
      parent.of.no$RemoveChild(selected.node$name)
      parent.of.no$AddChildNode(subtree)
    } else {
      tree.new <- subtree
    }
    
    break
  }
  if (!ok)
    return(NULL)
  return(list(
    tree.new=tree.new,
    l.prob=0,
    l.prob.rev=0
  ))
}

CART.move.swap <- function(tree) {
  if (tree$leafCount < 3) {
    return(NULL)
  }
  tree.new <- Clone(tree)
  split.node <- Traverse(tree.new, filterFun = function(node) {
    isNotLeaf(node) && (isNotLeaf(node$children[[1]]) || isNotLeaf(node$children[[2]]))
  })
  node.level <- sapply(split.node, function(node) node$level)
  node.prob <- as.probability(node.level**tree.new$param.split[2])
  
  if (length(split.node) == 0)
    return(NULL)
  ok <- F
  for (max.try in 1:1000) {
    selected.node.idx <- sample(1:length(node.prob), 1)#, prob = node.prob)
    selected.node <- split.node[[selected.node.idx]]
    rule.parent <- selected.node$split.rule
    
    subtree <- Clone(selected.node)
    subtree$full.obs <- tree.new$full.obs
    
    child.isLeaf <- sapply(subtree$children, isNotLeaf)
    if (all(child.isLeaf) && CART.compare.rule(subtree$children[[1]]$split.rule, subtree$children[[2]]$split.rule)) {
      rule.child <- subtree$children[[1]]$split.rule
      CART.update.rule.swap(subtree$children[[1]], rule.parent)
      CART.update.rule.swap(subtree$children[[2]], rule.parent)
    } else {
      child.node <- subtree$children[[sample(1:2, 1, prob = as.probability(child.isLeaf))]]
      rule.child <- child.node$split.rule
      CART.update.rule.swap(child.node, rule.parent)
    }
    
    if (!CART.update.rule(subtree, rule.child) || !CART.update.obs(subtree)) {
      next
    }
    ok <- T
    if (isNotRoot(selected.node)) {
      subtree$full.obs <- NULL
      parent.of.no <- selected.node$parent
      parent.of.no$RemoveChild(selected.node$name)
      parent.of.no$AddChildNode(subtree)
    } else {
      tree.new <- subtree
    }
    
    break
  }
  if (!ok)
    return(NULL)
  return(list(
    tree.new=tree.new,
    l.prob=0,
    l.prob.rev=0
  ))
}


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

#Parameters for prior
mc.params <- CART.extract.params(Sales~., Carseats)

#Record tree performance
criteria.name <- c("l.post", "SSE", "n.leaf")
criteria.func <- function(tree) {
  c(CART.prob.likelihood.regression(tree, mc.params)+CART.prob.prior(tree), CART.get.tree.SSE(tree), tree$leafCount)
}

#Initialize MCMC
mtree <- CART.create(Carseats, c(0.5, 1.5))
SSE0 <- CART.get.tree.SSE(mtree)
print(SSE0)
print(CART.prob.likelihood.regression(mtree, mc.params))

#MCMC
max.iter <- 500
criteria.matrix <- `colnames<-`(matrix(0, ncol=length(criteria.name), nrow=max.iter+1), criteria.name)
criteria.matrix[1,] <- criteria.func(mtree)

a.count <- 0
vec.moves <- c("grow", "prune", "change")
for (iter in 1:max.iter) {
  mc.move <- sample(vec.moves, 1)
  if (mc.move == "grow") {
    mc.new <- CART.move.grow(mtree)
  } else if (mc.move == "prune") {
    mc.new <- CART.move.prune(mtree)
  } else if (mc.move == "change") {
    mc.new <- CART.move.change(mtree)
  } else if (mc.move == "swap") {
    #Not implemented
  }
  
  if (is.null(mc.new) || !CART.check.tree.ok(mc.new$tree.new)) {
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
  
  if (iter %% 100 == 0) {
    print(paste0('============= #', as.character(iter), '(', format(a.count/iter, digits = 3),')'), quot=F)
    print(CART.prob.likelihood.regression(mtree, mc.params), quot=F)
    print(SSE0-CART.get.tree.SSE(mtree), quot=F)
  }
}

criteria.df <- as.data.frame(criteria.matrix)
ggarrange(ggplot(criteria.df) + geom_line(aes(x=0:max.iter, y=l.post))+xlab('iter')+theme_classic(),
          ggplot(criteria.df) + geom_line(aes(x=0:max.iter, y=SSE))+xlab('iter')+theme_classic(),
          ggplot(criteria.df) + geom_line(aes(x=0:max.iter, y=n.leaf))+xlab('iter')+theme_classic(),
          ncol = 1)

print(mtree)

##################### classification tree #####################

#Data loading
cancer <- read.csv('./data.csv')
cancer$diagnosis <- as.factor(cancer$diagnosis)
train.data <- cancer[,-c(1,33)]

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
max.iter <- 500
criteria.matrix <- `colnames<-`(matrix(0, ncol=length(criteria.name), nrow=max.iter+1), criteria.name)
criteria.matrix[1,] <- criteria.func(mtree)

a.count <- 0
vec.moves <- c("grow", "prune", "change")
for (iter in 1:max.iter) {
  mc.move <- sample(vec.moves, 1)
  if (mc.move == "grow") {
    mc.new <- CART.move.grow(mtree)
  } else if (mc.move == "prune") {
    mc.new <- CART.move.prune(mtree)
  } else if (mc.move == "change") {
    mc.new <- CART.move.change(mtree)
  } else if (mc.move == "swap") {
    
  }
  
  if (is.null(mc.new) || !CART.check.tree.ok(mc.new$tree.new)) {
    criteria.matrix[iter+1,] <- criteria.matrix[iter,]
    next
  }
  
  l.a.ratio.u <- (mc.new$l.prob.rev+CART.prob.likelihood.category(mc.new$tree.new, mc.params)+CART.prob.prior(mc.new$tree.new))
  l.a.ratio.l <- (mc.new$l.prob+CART.prob.likelihood.category(mtree, mc.params)+CART.prob.prior(mtree))
  if (log(runif(1)) <= (l.a.ratio.u-l.a.ratio.l)) {
    mtree <- mc.new$tree.new
    a.count <- a.count + 1
  }
  
  criteria.matrix[iter+1,] <- criteria.func(mtree)
  
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

print(mtree, "SSE", "obs.mean")


##################### MCMC functions #####################

CART.check.tree.ok <- function(tree) {
  all(sapply(tree$leaves, function(node) {length(node$obs.idx) > 0}))
}

CART.get.tree.SSE <- function(tree) {
  sum(sapply(tree$leaves, function(node) {
    obs.idx <- node$obs.idx
    if (any(is.na(obs.idx)) || length(obs.idx) == 0) {
      return(0)
    } else {
      y <- CART.get.obs(node)[,1]
      return(CART.get.data.SSE(y))
    }
  }))
}

CART.get.data.SSE <- function(data) {
  if (length(data) > 1) {
    return(var(data)*(length(data)-1))
  }
  return(0)
}

CART.get.tree.miscl <- function(tree) {
  n.total <- nrow(tree$full.obs)
  sum(sapply(tree$leaves, function(node) {
    obs.idx <- node$obs.idx
    if (any(is.na(obs.idx)) || length(obs.idx) == 0) {
      return(0)
    } else {
      y <- CART.get.obs(node)[,1]
      return(CART.get.data.miscl(y)*length(obs.idx)/n.total)
    }
  }))
}

CART.get.data.miscl <- function(data) {
  p <- as.numeric(table(data))
  p <- p/sum(p)
  1-max(p)
}

CART.extract.params <- function(formula., data, is.categorical=F) {
  if (is.categorical) {
    Y <- data[, all.vars(formula.)[1]]
    if (F) {
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

CART.prob.likelihood.regression <- function(tree, params, verb=F) {
  c.a <- params$a
  c.b <- tree$leafCount
  c.c <- ifelse(is.null(params$c), 0, as.numeric(params$c))
  c.mu <- params$mu
  c.nu <- params$nu
  c.lambda <- params$lambda
  
  l.prob <- c.c+0.5*c.b*log(c.a)-0.5*sum(sapply(tree$leaves, function(node) log(length(node$obs.idx))))-
    0.5*c.nu*c.lambda*log(sum(sapply(tree$leaves, function(node) {
      node.obs <- CART.get.obs(node)
      n.i <- nrow(node.obs)
      y.i <- node.obs[,1]
      s.i <- CART.get.data.SSE(y.i)
      t.i <- (mean(y.i)-c.mu)**2*c.a*n.i/(c.a+n.i)
      return(t.i+s.i)
    }))+c.nu*c.lambda)
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

CART.prob.select.rule <- function(obs, rule.values, is.categorical) {
  if (is.categorical) {
    return(-log(2**length(rule.values)-2)-log(ncol(obs)-1))
  } else {
    return(-log(length(rule.values))-log(ncol(obs)-1))
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
          split.value=rule.value,
          l.prob=CART.prob.select.rule(obs, rule.values, T)
        )
      )
    } else {
      rule.values <- sort(obs[,rule.colname], decreasing = T)[-1] #최대값은 빈 노드를 만들 수 있음.
      len.rule.values <- length(rule.values)
      if (len.rule.values <= 1 || rule.values[1] == rule.values[len.rule.values]) {
        col.candidates <- col.candidates[col.candidates != rule.colname]
        next
      }
      rule.value <- sample(rule.values, 1)
      return(
        list(
          split.col=rule.colname,
          split.value=rule.value,
          l.prob=CART.prob.select.rule(obs, rule.values, F)
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
  if (length(node$children) != length(child.names)) {
    print(node)
    print(length(node$children))
    print(length(child.names))
    print(child.names)
  }
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
    l.prob=log(node.prob[selected.node.idx])+rule$l.prob,
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
  l.prob.sel.split <- selected.node$split.rule$l.prob
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
    selected.node.idx <- sample(1:length(node.prob), 1, prob = node.prob)
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





#
# Generate random values from a mixture distribution.
#
rmix <- function(n, mu, sigma, p) {
  matrix(rnorm(length(mu)*n, mu, sigma), ncol=n)[
         cbind(sample.int(length(mu), n, replace=TRUE, prob=p), 1:n)]
}
mu <- c(25, 60, 130, 190) # Means
sigma <- c(8, 13, 15, 19) # SDs
p <- c(.18, .2, .24, .28) # Relative proportions (needn't sum to 1)
n <- 1e4                  # Sample size
x <- rmix(n, mu, sigma, p)
#
# Find the modes of a KDE.
# (Quick and dirty: it assumes no mode spans more than one x value.)
#
findmodes <- function(kde) {
  kde$x[which(c(kde$y[-1],NA) < kde$y & kde$y > c(NA,kde$y[-length(kde$y)]))]
}
#
# Compute the mode trace by varying the bandwidth within a factor of 10 of
# the default bandwidth.  Track the modes as the bandwidth is decreased from
# its largest to its smallest value.
# This calculation is fast, so we can afford a detailed search.
#
m <- mean(x)
id <- 1
bw <- density(x)$bw * 10^seq(1,-1, length.out=101) 
modes.lst <- lapply(bw, function(h) {
  m.new <- sort(findmodes(density(x, bw=h)))
  # -- Associate each previous mode with a nearest new mode.
  if (length(m.new)==1) delta <- Inf else delta <- min(diff(m.new))/2
  d <- outer(m.new, m, function(x,y) abs(x-y))
  i <- apply(d, 2, which.min)
  g <- rep(NA_integer_, length(m.new))
  g[i] <- id[1:ncol(d)]
  #-- Create new ids for new modes that appear.
  k <- is.na(g)
  g[k] <- (sum(!k)+1):length(g)
  id <<- g
  m <<- m.new
  data.frame(bw=h, Mode=m.new, id=g)
})

myf <- function(h) {
    sort(findmodes(density(x, bw=h)))
}

myg <- function(h) {
    m.new <- sort(findmodes(density(x, bw=h)))
    if (length(m.new)==1) delta <- Inf else delta <- min(diff(m.new))/2
    d <- outer(m.new, m, function(x,y) abs(x-y))
    print(paste(':', nrow(d), ncol(d)))
}

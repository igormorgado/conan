myf <- function(h) {
  m.new <- sort(findmodes(density(x, bw=h)))
  # -- Associate each previous mode with a nearest new mode.
  if (length(m.new)==1) delta <- Inf else delta <- min(diff(m.new))/2
  d <- outer(m.new, m, function(x,y) abs(x-y))
  print(d)
  i <- apply(d, 2, which.min)
  print(i)
  g <- rep(NA_integer_, length(m.new))
  g[i] <- id[1:ncol(d)]
  #-- Create new ids for new modes that appear.
  k <- is.na(g)
  g[k] <- (sum(!k)+1):length(g)
  id <<- g
  m <<- m.new
  data.frame(bw=h, Mode=m.new, id=g)
}

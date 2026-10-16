# Run from repository root: Rscript tests/generate_ggir_fixture.R /path/to/pinned/HASIB.R
source(commandArgs(trailingOnly=TRUE)[1])
# HASIB's own helpers, extracted without reimplementing their mathematics.
f=HASIB
body(f)=as.call(c(as.list(body(f))[1:5],quote(list(sum=sumPerWindow,roll=create_rollfun_mat))))
h=f()
cases=list(quiet=rep(0,600), active=rep(4,600), varying=rep(c(0,0,1,2,0,8,0,0,3,1),each=60),
           missing=c(rep(0,200),NA,rep(3,399)),partial=c(rep(c(0,3,1,0),each=150),rep(2,17)))
for (j in c(1,5,8,20)) {
 x=rep(0,600);x[((j-1)*30+1):(j*30)]=20
 cases[[paste0('impulse',j)]]=x
}
inputs=outputs=NULL
for (name in names(cases)) {
 x=cases[[name]]
 inputs=rbind(inputs,data.frame(case=name,index=seq_along(x)-1,count=x))
 # g.sib.det.R fixes missing values before calling HASIB, zc.scale defaults to 1.
 x[is.na(x)]=0
 activity=h$sum(x,2,60)/6
 scores=as.vector(.001 * (h$roll(activity,7) %*% c(67,74,230,76,58,54,106)))
 scores=c(rep(2,4),scores)
 full=HASIB(HASIB.algo='ColeKripke1992',time=seq_along(x),ws3=2,zeroCrossingCount=x)
 starts=seq(1,length(x),by=30)
 outputs=rbind(outputs,data.frame(case=name,minute=seq_along(starts)-1,
                    wake=1-full[starts,1],score=scores[seq_along(starts)]))
}
dir.create('tests/fixtures',showWarnings=FALSE)
write.csv(inputs,'tests/fixtures/ggir_counts.csv',row.names=FALSE,na='NA')
write.csv(outputs,'tests/fixtures/ggir_states.csv',row.names=FALSE,na='NA')

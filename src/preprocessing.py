def get_gene_length(gene_id, db):
    exon_intervals=[]
    
  
    
    gene = db[gene_id]
    for exon in db.children(gene, featuretype='exon', level=2):
        exon_intervals.append([exon.start, exon.end])
            
    
    exon_intervals.sort()
    
    collapsed_exons=[]
    
    for start, end in exon_intervals:
        if not collapsed_exons:
            collapsed_exons.append([start,end])
        else:
            last_start, last_end = collapsed_exons[-1]
            if start<=last_end:
                collapsed_exons[-1][1]= max(last_end, end)
            else:
                collapsed_exons.append([start,end])
    
    
    
    
    total_length = sum(end-start+1 for start, end in collapsed_exons)
    
    return total_length
                
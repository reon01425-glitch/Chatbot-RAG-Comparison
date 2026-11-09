import csv,re,sys
src='audit_sheet.csv'; ans=open('audit_sheet_answers.txt',encoding='utf-8').read()
A={(m.group(1).strip(),m.group(2).strip()):m.group(3).strip() for m in re.finditer(r'^=== (.+?) / (.+?) ===\n(.*?)(?=^=== |\Z)',ans,re.S|re.M)}
rows=list(csv.DictReader(open(src,encoding='utf-8-sig')))
keys=list(rows[0].keys()); sysk=keys[0]
out=csv.writer(open('audit_review/review_rows_15_60.csv','w',newline='',encoding='utf-8-sig'))
out.writerow(['row','system','qid','step_n','gold_actor','gold_also_ok','gold_keywords','system_answer','human_detected (y/n)','human_actor_status (correct/mismatch/unattributed/not_detected)','catatan'])
miss=0
for i,r in enumerate(rows,1):
    if i<15: continue
    a=A.get((r[sysk],r['qid']));miss+=a is None
    out.writerow([i,r[sysk],r['qid'],r['step_n'],r['gold_actor'],r['gold_also_ok'],r['gold_keywords'],a or '[ANSWER NOT FOUND]','','',''])
print('written',len(rows)-14,'rows; missing answers:',miss)

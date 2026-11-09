"""Copy reviewed labels from review_rows_15_60.csv into audit_sheet.csv (backs up first). Refuses if any row is blank/invalid."""
import csv,shutil
rev=list(csv.reader(open('audit_review/review_rows_15_60.csv',encoding='utf-8-sig')))[1:]
D={'y','n'};S={'correct','mismatch','unattributed','not_detected'};bad=[r[0] for r in rev if r[8].strip() not in D or r[9].strip() not in S]
if bad: raise SystemExit('Blank/invalid rows: '+', '.join(bad))
shutil.copy('audit_sheet.csv','audit_sheet.before_merge.csv')
rd=list(csv.reader(open('audit_sheet.csv',encoding='utf-8-sig')))
for r in rev:
    row=rd[int(r[0])]; row[10],row[11],row[12]=r[8].strip(),r[9].strip(),r[10]
w=csv.writer(open('audit_sheet.csv','w',newline='',encoding='utf-8-sig'),quoting=csv.QUOTE_ALL); w.writerows(rd)
print('merged',len(rev),'rows; backup at audit_sheet.before_merge.csv')

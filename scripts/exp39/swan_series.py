import struct, sys
b=open(sys.argv[1],"rb").read()
def varint(b,i):
    v=s=0
    while True:
        c=b[i]; v|=(c&0x7f)<<s; i+=1; s+=7
        if c<0x80: return v,i
def series(key):
    k=b"\n"+bytes([len(key)])+key.encode()+b"\x10"; out=[]; i=0
    while True:
        i=b.find(k,i)
        if i<0: return out
        step,j=varint(b,i+len(k))
        m=b.find(b"*\t\t",j,j+60)
        if m>0: out.append((step, round(struct.unpack("<d",b[m+3:m+11])[0],4)))
        i=j
for key in sys.argv[2:]:
    print(key, series(key))

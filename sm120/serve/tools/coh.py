"""coherence probe: 'capital of France, one word' -> Paris; 300-token story, no loops (distinct 8-gram ratio). Usage: coh.py [label]"""
import json,re,sys,urllib.request
K=re.search(r'VLLM_API_KEY=(\S+)',open('/home/jarrelscy/homeassistant/.env').read()).group(1)
H={'Authorization':'Bearer '+K,'Content-Type':'application/json'}
def chat(p,n):
    r=json.load(urllib.request.urlopen(urllib.request.Request('http://localhost:8001/v1/chat/completions',headers=H,data=json.dumps(dict(
        model='local',messages=[{'role':'user','content':p}],max_tokens=n,temperature=0,chat_template_kwargs=dict(enable_thinking=False))).encode())))
    return r['choices'][0]['message']['content'] or ''
a=chat('What is the capital of France? Answer in one word.',512)
s=chat('Write a short story (about 250 words) about a lighthouse keeper who finds a message in a bottle.',300)
w=s.split();g=[tuple(w[i:i+8]) for i in range(max(0,len(w)-7))];dr=len(set(g))/max(1,len(g))
ok=('paris' in a.lower()) and len(w)>120 and dr>0.9
print(json.dumps(dict(label=sys.argv[1] if len(sys.argv)>1 else '',pass_=ok,paris=a.strip()[-60:],story_words=len(w),distinct8=round(dr,3),story_head=s[:160])))

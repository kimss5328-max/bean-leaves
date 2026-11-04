from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
import ai.services as services
from main import app

def test_sync(monkeypatch):
    monkeypatch.setattr(services, 'get_classifier', lambda: (lambda image, top_k=5:
        [{'label':'object','score':0.9}]))
    buf=BytesIO(); Image.new('RGB',(32,32),'blue').save(buf,format='PNG')
    with TestClient(app) as c:
        assert c.get('/health').status_code==200
        assert c.get('/vision/sync/').status_code==200
        assert c.get('/static/style.css').status_code==200
        r=c.post('/api/sync/classify/',files={'image':('test.png',buf.getvalue(),'image/png')})
        assert r.status_code==200 and r.json()['top1']['label']=='object'
        assert c.post('/api/sync/classify/',files={'image':('bad.txt',b'bad','text/plain')}).status_code==400
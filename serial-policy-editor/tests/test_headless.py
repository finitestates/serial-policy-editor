import json
import threading
import urllib.request
import urllib.error
from http.server import HTTPServer

import pytest

from tests.test_episode_runtime import NoEogBackend
from trajectory_editor.headless import Session, Conflict, make_handler
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.domain import EditorError


def send(session, operation, **payload):
    return session.mutate(operation, {"request_id": str(session.sequence), "revision": session.revision, **payload})


@pytest.fixture
def session(tmp_path):
    backend = NoEogBackend()
    with EpisodeStore(tmp_path / 'test.sqlite3') as store:
        yield Session(backend, store, backend.provenance())


def test_edit_fork_rewind_reopen_preserves_evidence_and_budget(session):
    send(session, 'open', prompt='P')
    parent = session.episode_id
    send(session, 'settings', max_tokens=8, sampling={'temperature': 0})
    send(session, 'actions', action={'kind':'select-raw-rank', 'rank':2})
    send(session, 'actions', action={'kind':'write','text':' A B','mode':'exact'})
    send(session, 'actions', action={'kind':'hold','limit':2})
    assert session.engine.boundary == 5
    assert len(session.store.actions(parent)) == 3
    send(session, 'fork', boundary=2)
    child = session.episode_id
    assert child != parent
    assert session.engine.boundary == 0
    assert session.engine.remaining == 6
    assert session.store.get_episode(child)['parent_episode_id'] == parent
    send(session, 'actions', action={'kind':'hold','limit':2})
    send(session, 'rewind', boundary=1)
    tokens = list(session.engine.token_ids)
    sampling = session.engine.sampling
    send(session, 'close')
    send(session, 'open', episode_id=child)
    assert session.engine.token_ids == tokens
    assert session.engine.sampling == sampling
    assert session.engine.remaining == 5
    assert len(session.store.tokens(child)) == 1
    assert len(session.store.tokens(parent)) == 5


def test_receipt_prevents_duplicate_action_and_rejects_reuse(session):
    send(session, 'open', prompt='P')
    p={'request_id':'pick', 'revision':session.revision,'action':{'kind':'accept'}}
    result=session.mutate('actions',p)
    assert session.mutate('actions',p)==result
    assert session.engine.boundary==1
    with pytest.raises(Conflict):
        session.mutate('actions',{**p,'action':{'kind':'hold','limit':2}})
    with pytest.raises(Conflict):
        session.mutate('actions',{**p,'request_id':'stale'})
    assert session.engine.boundary==1


def test_sampler_changes_invalidate_observation_without_renewing_budget(session):
    send(session,'open',prompt='P')
    send(session,'settings',max_tokens=5)
    send(session,'actions',action={'kind':'accept'})
    old=session.observation()
    send(session,'settings',sampling={'temperature':0})
    assert session.engine.remaining==4
    assert session.observation()['revision']!=old['revision']


def test_invalid_and_unbounded_actions_do_not_write_evidence(session):
    send(session,'open',prompt='P')
    for action in ({'kind':'hold','limit':257},{'kind':'finish'},{'kind':'select','rank':False}):
        with pytest.raises(EditorError):
            send(session,'actions',action=action)
    assert not session.store.actions(session.episode_id)
    assert session.engine.boundary==0


def test_end_seals_without_generating_and_allows_fork(session):
    send(session,'open',prompt='P')
    send(session,'end')
    assert session.engine.ended
    assert session.engine.boundary==0
    assert session.store.get_episode(session.episode_id)['status']=='completed'
    send(session,'fork',boundary=0)
    assert not session.engine.ended


def test_model_mismatch_is_rejected_before_backend_is_repositioned(session):
    send(session,'open',prompt='P')
    saved=session.episode_id
    session.provenance={'backend':'different'}
    tokens=list(session.backend.tokens)
    with pytest.raises(EditorError,match='different model'):
        send(session,'open',episode_id=saved)
    assert session.backend.tokens==tokens


def test_http_journey_and_origin_protection(tmp_path):
    ready=threading.Event()
    holder=[]
    def run():
        backend=NoEogBackend()
        with EpisodeStore(tmp_path/'http.sqlite3') as store:
            with HTTPServer(('127.0.0.1',0),make_handler(Session(backend,store,backend.provenance()))) as server:
                holder.append(server)
                ready.set()
                server.serve_forever()
    worker=threading.Thread(target=run,daemon=True)
    worker.start()
    assert ready.wait(5)
    server=holder[0]
    base=f'http://127.0.0.1:{server.server_port}'
    def request(path, payload=None, headers=None):
        data=None if payload is None else json.dumps(payload).encode()
        req=urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json',**(headers or {})})
        with urllib.request.urlopen(req,timeout=5) as response:
            return response.read()
    try:
        assert b'Serial Policy Editor' in request('/')
        state=json.loads(request('/api/session'))
        payload={'request_id':'create','revision':state['revision'],'prompt':'P'}
        result=json.loads(request('/api/session',payload))
        assert result['boundary']==0
        assert json.loads(request('/api/session',payload))==result
        assert json.loads(request('/api/observation'))['candidates']
        assert json.loads(request('/api/episodes'))['episodes'][0]['initial_text']=='P'
        for headers in ({'Origin':'https://example.com'},{'Host':'attacker.example'}):
            with pytest.raises(urllib.error.HTTPError) as exc:
                request('/api/session',headers=headers)
            assert exc.value.code==403
        with pytest.raises(urllib.error.HTTPError) as exc:
            request('/api/session',{**payload,'request_id':'stale'})
        assert exc.value.code==409
    finally:
        server.shutdown()
        worker.join(5)

"""Region feedback must change geometry, survive regeneration, and report failures."""
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from dot2dot.feedback import Correction, apply_feedback, check_feedback
from dot2dot.models import Path
from dot2dot.pipeline import assemble
from dot2dot.web.app import app


def correction(note='Restore the missing eye', action='auto', box=(0.3,0.3,0.7,0.7), id='eye'):
    return Correction(id=id, note=note, action=action, box=box)


def test_normalized_regions_and_negated_removal():
    assert correction().pixels((200,400,3)) == (120,60,280,140)
    assert correction("Don't remove the eye").resolved_action() == 'protect'
    assert correction('Remove these lines').resolved_action() == 'remove'
    assert correction('Too busy; simplify').resolved_action() == 'simplify'
    for box in [(0.8,0.2,0.1,0.5), (0,0,float('nan'),1),(-0.1,0,1,1)]:
        with pytest.raises(ValidationError):
            correction(box=box)


def test_restore_recovers_missing_eye_from_source_and_reports_coverage():
    image = np.full((200,200,3),255,np.uint8)
    cv2.ellipse(image,(100,100),(20,10),0,0,360,(0,0,0),2)
    outer = Path(np.array([[10.,10],[190,10],[190,190],[10,190]]),True,essential=True)
    regions=[correction()]
    paths, affected=apply_feedback([outer],regions,image)
    assert outer in paths and affected['eye'] > 0
    assert any(p.protected and not p.essential for p in paths)
    puzzle=assemble(paths,200,200,80)
    checks=check_feedback(puzzle,paths,regions,affected)
    assert checks[0].coverage > 0.8
    assert checks[0].status == 'geometry_checked'


def test_remove_clips_only_marked_geometry_without_reconnecting_it():
    image=np.full((200,200,3),255,np.uint8)
    line=Path(np.array([[float(x),100] for x in range(10,190)]),False)
    regions=[correction('Remove this line','remove')]
    paths,affected=apply_feedback([line],regions,image)
    assert len(paths)==2
    assert all(not ((p.points[:,0]>=60)&(p.points[:,0]<140)).any() for p in paths)
    puzzle=assemble(paths,200,200,20)
    checks=check_feedback(puzzle,paths,regions,affected)
    assert checks[0].status=='geometry_checked'


def test_blank_region_is_not_reported_as_fixed():
    image=np.full((200,200,3),255,np.uint8)
    outer=Path(np.array([[10.,10],[190,10],[190,190],[10,190]]),True,essential=True)
    regions=[correction()]
    paths,affected=apply_feedback([outer],regions,image)
    puzzle=assemble(paths,200,200,20)
    check=check_feedback(puzzle,paths,regions,affected)[0]
    assert check.status=='needs_review' and check.coverage is None


def test_accumulated_regions_are_reapplied_and_global_numbers_are_contiguous():
    image=np.full((200,200,3),255,np.uint8)
    cv2.circle(image,(50,50),12,(0,0,0),2)
    cv2.circle(image,(150,150),12,(0,0,0),2)
    regions=[correction(box=(.15,.15,.35,.35),id='left'),correction(box=(.65,.65,.85,.85),id='right')]
    outer=Path(np.array([[10.,10],[190,10],[190,190],[10,190]]),True,essential=True)
    first,counts=apply_feedback([outer],regions,image)
    second,counts2=apply_feedback([outer],regions,image)
    assert counts==counts2 and counts['left']>0 and counts['right']>0
    puzzle=assemble(second,200,200,80)
    assert [d.number for d in puzzle.dots]==list(range(1,len(puzzle.dots)+1))


def upload(client,path,**data):
    with open(path,'rb') as file:
        return client.post('/api/puzzles',files={'image':('source.png',file)},data=data)


def test_web_feedback_rebuilds_puzzle_pdf_and_returns_checks(square_image):
    client=TestClient(app)
    baseline=upload(client,square_image,style='lineart',dots='50').json()
    annotations=[correction('Remove the top boundary','remove',box=(.2,.2,.8,.3)).model_dump()]
    response=upload(client,square_image,style='lineart',dots='50',feedback=json.dumps(annotations))
    assert response.status_code==200,response.text
    revised=response.json()
    assert revised['id']!=baseline['id']
    assert revised['puzzle_svg']!=baseline['puzzle_svg']
    assert revised['corrections']==json.loads(json.dumps(annotations))
    assert revised['correction_checks'][0]['action']=='remove'
    assert client.get(revised['pdf_url']).content.startswith(b'%PDF')


def test_web_rejects_bad_feedback_and_falls_back_when_ai_unavailable(square_image,monkeypatch):
    client=TestClient(app)
    for raw in ['bad','{}',json.dumps([correction().model_dump()]*2)]:
        assert upload(client,square_image,feedback=raw).status_code==400
    monkeypatch.setattr('dot2dot.web.app.get_planner',lambda:None)
    result=upload(client,square_image,style='lineart',feedback=json.dumps([correction().model_dump()]),feedback_ai='true')
    assert result.status_code==200
    assert not result.json()['feedback_ai_used']
    assert any('unavailable' in w for w in result.json()['warnings'])


def test_ai_interpretation_receives_only_marked_crop_and_note(square_image,monkeypatch):
    from dot2dot.planner import Plan,Part
    calls=[]
    class Planner:
        def plan(self,image,request):
            calls.append((image.shape,request))
            return Plan('eye','cartoon','lineart',50,[Part('edge',(0.1,0.1,0.9,0.9))])
    monkeypatch.setattr('dot2dot.web.app.get_planner',lambda:Planner())
    client=TestClient(app)
    regions=[correction('Keep the eye clear','protect',box=(.2,.2,.5,.5)).model_dump()]
    response=upload(client,square_image,style='lineart',feedback=json.dumps(regions),feedback_ai='true')
    assert response.status_code==200,response.text
    assert response.json()['feedback_ai_used']
    assert len(calls)==1
    assert calls[0][0][:2]==(round(response.json()['height']*.3),round(response.json()['width']*.3))
    assert 'Keep the eye clear' in calls[0][1]


def test_later_overlapping_remove_wins_over_restore():
    image=np.full((200,200,3),255,np.uint8)
    cv2.circle(image,(100,100),15,(0,0,0),2)
    regions=[correction(id='restore'),correction('Remove these lines','remove',id='remove')]
    paths,_=apply_feedback([],regions,image)
    assert not paths

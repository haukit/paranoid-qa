from paranoid_qa.contracts.aggregate import AggregateAnswer
from paranoid_qa.contracts.common import SourceRef
from paranoid_qa.contracts.specific import SpecificAnswer
from paranoid_qa.contracts.state import GraphState


def test_sourceref_formatting():
    assert str(SourceRef(document_id="a", filename="a.pdf", page="3")) == "a.pdf p.3"
    assert str(SourceRef(document_id="a", filename="a.pdf")) == "a.pdf"


def test_state_accepts_either_answer_type():
    specific: GraphState = {"answer": SpecificAnswer()}
    aggregate: GraphState = {"answer": AggregateAnswer()}
    assert isinstance(specific["answer"], SpecificAnswer)
    assert isinstance(aggregate["answer"], AggregateAnswer)

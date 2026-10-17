import pytest


def test_worm_filters_muscles_by_explicit_allowlist_and_deduplicates_electrical():
    from neurograft.acquire import normalize_worm_csv
    payload = b'Source,Target,Weight,Type\nA,B,3,chemical\nA,muscle,4,chemical\nA,B,2,electrical\nB,A,2,electrical\n'
    rows, receipt = normalize_worm_csv(payload, {'A', 'B'})
    assert len(rows) == 2
    assert rows[0] == {'src': 'A', 'dst': 'B', 'relation': 'chemical', 'weight': 3.0, 'directed': True}
    assert rows[1]['directed'] is False
    assert receipt['excluded_non_neuron_rows'] == 1
    assert receipt['deduplicated_electrical_rows'] == 1


@pytest.mark.parametrize('weight', ['nan', 'inf', '-1', '0', '1.5'])
def test_worm_rejects_invalid_synapse_counts(weight):
    from neurograft.acquire import normalize_worm_csv
    with pytest.raises(ValueError, match='weight'):
        normalize_worm_csv(f'Source,Target,Weight,Type\nA,B,{weight},chemical\n'.encode(), {'A', 'B'})


def test_worm_conflicting_electrical_multiplicity_is_not_silently_summed():
    from neurograft.acquire import normalize_worm_csv
    with pytest.raises(ValueError, match='conflicting'):
        normalize_worm_csv(b'Source,Target,Weight,Type\nA,B,2,electrical\nB,A,3,electrical\n', {'A', 'B'})


def test_neuron_allowlist_uses_declared_network_populations_and_aligns_csv_aliases():
    from neurograft.acquire import neuron_ids_from_network, align_worm_neurons
    xml = b'<neuroml xmlns="http://www.neuroml.org/schema/neuroml2"><network id="CellsOnly"><population id="VA1" component="VA1" size="1"/><population id="CANL" component="CANL" size="1"/></network></neuroml>'
    ids = neuron_ids_from_network(xml)
    assert ids == {'VA1', 'CANL'}
    allowed, aliases = align_worm_neurons(b'Source,Target,Weight,Type\nVA01,muscle,3,chemical\n', ids)
    assert allowed == {'VA01', 'CANL'}
    assert aliases == {'VA01': 'VA1'}


def test_source_download_refuses_mutable_revision_before_network(tmp_path):
    from neurograft.acquire import fetch_worm
    with pytest.raises(ValueError, match='revision'):
        fetch_worm(tmp_path, revision='master')


def test_fetch_retains_raw_network_and_uses_explicit_neuron_population_ids(tmp_path, monkeypatch):
    import neurograft.acquire as acquire
    def reader(url, max_bytes):
        return (b'<neuroml><network><population id="A" size="1"/><population id="B" size="1"/></network></neuroml>' if url.endswith('.net.nml') else
                b'Source,Target,Weight,Type\nA,B,3,chemical\n')
    monkeypatch.setattr(acquire, '_download', reader)
    receipt = acquire.fetch_worm(tmp_path)
    assert receipt['declared_neurons'] == 2
    assert (tmp_path / 'worm.bundle.json').exists()
    assert (tmp_path / 'CellsOnly.net.nml').exists()


def test_conflicting_pairs_can_be_explicitly_excluded_with_receipt():
    from neurograft.acquire import normalize_worm_csv
    raw = b'Source,Target,Weight,Type\nA,B,3,chemical\nA,B,2,electrical\nB,A,3,electrical\n'
    rows, receipt = normalize_worm_csv(raw, {'A', 'B'}, electrical_conflicts='exclude')
    assert len(rows) == 1 and rows[0]['relation'] == 'chemical'
    assert receipt['conflicting_electrical_pairs'] == [{'src': 'A', 'dst': 'B', 'weights': [2.0, 3.0]}]
    assert receipt['excluded_conflicting_electrical_rows'] == 2

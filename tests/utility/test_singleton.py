from recwizard.utils import Singleton
import torch


# Tests
def test_singleton_creates_instance():
    uid = "test_uid"
    instance = torch.nn.Module()
    result = Singleton(uid, instance)
    assert result is instance


def test_singleton_reuses_instance():
    uid = "test_uid"
    instance1 = torch.nn.Module()
    instance2 = torch.nn.Module()

    result1 = Singleton(uid, instance1)
    result2 = Singleton(uid, instance2)

    assert type(result1) is type(instance1)
    assert type(result2) is type(instance1)
    assert result1.__dict__ == instance1.__dict__
    assert result2.__dict__ == instance1.__dict__


def test_singleton_instance_count():
    uid = "test_uid"
    instance1 = torch.nn.Module()
    instance2 = torch.nn.Module()

    result1 = Singleton(uid, instance1)
    result2 = Singleton(uid, instance2)

    assert type(result1) is type(instance1)
    assert type(result2) is type(instance1)
    assert result1.__dict__ == instance1.__dict__
    assert result2.__dict__ == instance1.__dict__


def test_singleton_different_uids():
    uid1 = "test_uid_1"
    uid2 = "test_uid_2"
    instance1 = torch.nn.Module()
    instance2 = torch.nn.Module()

    result1 = Singleton(uid1, instance1)
    result2 = Singleton(uid2, instance2)

    assert type(result1) is type(instance1)
    assert type(result2) is type(instance2)
    assert result1.__dict__ == instance1.__dict__
    assert result2.__dict__ == instance2.__dict__

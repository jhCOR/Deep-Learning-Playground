from collections.abc import Iterable

class ObjectHelper():
    @classmethod
    def is_NestedIterable(cls, target):
        if cls.isIterable(target) and all(cls.isIterable(target) for sublist in target):
            return True
        else:
            return False

    @classmethod
    def isIterable(self, target):
        return True if isinstance(target, Iterable) else False
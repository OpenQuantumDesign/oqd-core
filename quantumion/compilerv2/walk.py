from abc import abstractmethod

########################################################################################

from quantumion.compilerv2.base import PassBase

########################################################################################
from rich import print as pprint

class Walk(PassBase):
    def __init__(self, rule: PassBase):
        super().__init__()

        self.rule = rule
        pass

    @abstractmethod
    def walk_VisitableBaseModel(self, model):
        pass

    @property
    def children(self):
        return [self.rule]

    def map(self, model):
        return self.walk(model)

    def walk(self, model):
        for cls in model.__class__.__mro__:
            walk_func = getattr(self, "walk_{}".format(cls.__name__), None)
            if walk_func:
                break

        if not walk_func:
            walk_func = self.generic_walk

        return walk_func(model)

    def generic_walk(self, model):
        return self.rule(model)

    pass


########################################################################################


class Pre(Walk):
    def walk_dict(self, model):
        new_model = self.rule(model)

        new_model = {k: self(v) for k, v in new_model.items()}

        return new_model

    def walk_list(self, model):
        new_model = self.rule(model)

        new_model = [self(e) for e in new_model]

        return new_model

    def walk_tuple(self, model):
        new_model = self.rule(model)

        new_model = tuple([self(e) for e in new_model])

        return new_model

    def walk_VisitableBaseModel(self, model):
        new_model = self.rule(model)

        new_fields = {}
        for key in new_model.model_fields.keys():
            new_fields[key] = self(getattr(new_model, key))
        new_model = new_model.__class__(**new_fields)

        return new_model


class Post(Walk):
    def walk_dict(self, model):
        new_model = {k: self(v) for k, v in model.items()}

        new_model = self.rule(new_model)
        return new_model

    def walk_list(self, model):
        new_model = [self(e) for e in model]

        new_model = self.rule(new_model)

        return new_model

    def walk_tuple(self, model):
        new_model = tuple([self(e) for e in model])

        new_model = self.rule(new_model)

        return new_model

    def walk_VisitableBaseModel(self, model):
        new_fields = {}
        for key in model.model_fields.keys():
            new_fields[key] = self(getattr(model, key))
        new_model = model.__class__(**new_fields)

        new_model = self.rule(new_model)

        return new_model

class PreWithNonVisitableOutput(Walk):
    """For VisitableBaseMode, when self.rule(model) creates an object which is not a VisitableBaseMode, walk is applied again"""
    def walk_dict(self, model):
        new_model = self.rule(model)

        new_model = {k: self(v) for k, v in new_model.items()}

        return new_model

    def walk_list(self, model):
        new_model = self.rule(model)

        new_model = [self(e) for e in new_model]

        return new_model

    def walk_tuple(self, model):
        new_model = self.rule(model)

        new_model = tuple([self(e) for e in new_model])

        return new_model

    def walk_VisitableBaseModel(self, model):
        new_model = self.rule(model)

        if not isinstance(new_model, VisitableBaseModel):
            return self.walk(new_model)

        new_fields = {}
        for key in new_model.model_fields.keys():
            new_fields[key] = self(getattr(new_model, key))
        new_model = new_model.__class__(**new_fields)

        return new_model
class Wrapper(object):

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, attr):

        is_magic = attr.startswith('__') and attr.endswith('__')
        if is_magic:
            return super().__getattr__(attr)
        try:

            return self.__dict__[attr]
        except:

            return getattr(self.inner, attr)

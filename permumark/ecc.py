"""Generalized Reed-Solomon code for error correction using SageMath."""

from __future__ import annotations

from sage.all import GF, Integer, codes, matrix, sage, vector


def gf_element_to_int(w) -> int:
    """
    Convert a GF-element (with base 2) to an integer.
    :param w: an element from the finite field
    :return: the integer representation of the element
    """
    if w == 0:
        return 0
    return int(
        "".join(str(int(c)) for c in w.polynomial().coefficients(sparse=False)[::-1]), 2
    )


class ErrorCorrectionCode:
    """
    The abstract class for error correction codes.
    :param q: size of alphabet
    :param length: size of codeword
    :param dimension: size of message
    """

    def __init__(self, q: int, length: int, dimension: int) -> None:
        self.q = q
        self.length = length
        self.dimension = dimension

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.q}, {self.length}, {self.dimension})"

    def encode(self, message: list[int]) -> list[int]:
        """The encoding function.
        :param message: a list of elements in the alphabet
        :return the encoded codeword
        """
        raise NotImplementedError

    def decode(
        self, received: list[int], erasure_idx: list[int] | None = None
    ) -> list[int] | None:
        """The decoding function.
        :param received: a list of elements in the alphabet to decode
        :param erasure_idx: indices of erasure. Defaults to None
        :return the decoded message if successful, None otherwise
        """
        raise NotImplementedError


class ReedSolomonCode(ErrorCorrectionCode):
    """Generalized Reed-Solomon code.
    :param q: size of the finite field
    :param length: size of codeword
    :param dimension: size of message
    """

    def __init__(self, q: int, length: int, dimension: int) -> None:
        super().__init__(q, length, dimension)
        self.length = length
        self.dimension = dimension
        self.F = GF(Integer(q), "x")
        self.rs = codes.GeneralizedReedSolomonCode(
            # evaluation_points=self.F.list()[1 : length + 1],
            evaluation_points=[self.F.from_integer(i) for i in range(1, length + 1)],
            dimension=dimension,
            column_multipliers=[self.F.from_integer(i) for i in range(1, length + 1)],
        )
        self.decoder = self.rs.decoder("ErrorErasure")
        self.decoder = codes.decoders.GRSErrorErasureDecoder(self.rs)
        self.GF2 = GF(2)

    def encode(self, message: list[int]) -> list[int]:
        """The encoding function.
        :param message: a list of elements in the alphabet
        :return the encoded codeword
        """
        gf_message = vector(self.F, [self.F.from_integer(w) for w in message])
        gf_codeword = self.rs.encode(gf_message)
        return [gf_element_to_int(w) for w in gf_codeword]

    def decode(
        self, received: list[int], erasure_idx: list[int] | None = None
    ) -> list[int] | None:
        """The decoding function.
        :param received: a list of elements in the alphabet to decode
        :param erasure_idx: indices of erasure. Defaults to None
        :return the decoded message if successful, None otherwise
        """
        try:
            gf_received = vector(self.F, [self.F.from_integer(w) for w in received])
            if erasure_idx:
                if len(erasure_idx) == self.length - self.dimension:
                    # NOTE: this is a bug in sage, when number of erasures equals n - k
                    cut = self.length - len(erasure_idx)
                    indices = tuple(
                        i for i in range(self.length) if i not in erasure_idx
                    )
                    gf_received = vector(
                        self.F, [self.F.from_integer(received[i]) for i in indices]
                    )
                    eval_points = [self.F.from_integer(i + 1) for i in indices]
                    multipliers = [self.F.from_integer(i + 1) for i in indices]
                    vandermonde = matrix(
                        [
                            [multipliers[i] * eval_points[i] ** j for j in range(cut)]
                            for i in range(cut)
                        ]
                    )
                    gf_decoded = vandermonde.inverse() * gf_received
                    return [gf_element_to_int(w) for w in gf_decoded]
                erasure_vec = vector(
                    self.GF2,
                    [self.GF2(int(i in erasure_idx)) for i in range(len(received))],
                )
                gf_decoded = self.decoder.decode_to_message((gf_received, erasure_vec))
            else:
                gf_decoded = self.rs.decode_to_message(gf_received)
            return [gf_element_to_int(w) for w in gf_decoded]
        except sage.coding.decoder.DecodingError:
            return None

import struct

import numpy as np


class PointCloud:
    """
    PCD format to (x, y, z, intensity) data.
    Only binary-based PCD is supported.

    Use attribute 'data' to get the numpy array (float32).
    """

    def __init__(self, filename: str, use_intensity=True):
        self.fields = []
        self.formats = []
        self.sizes = []
        self.types = []
        self.counts = []
        self.viewpoint = []
        self.points = 0
        self.use_intensity = use_intensity

        with open(filename, 'rb') as f:
            lines = f.readlines()
        header = [line.decode("utf-8") for line in lines[:11]]
        binary_data = bytearray(b''.join(lines[11:]))

        self.parse_header(header)
        self.data = self.parse_binary_data(binary_data)

    def parse_header(self, header):
        version = header[1].strip()
        assert version == 'VERSION 0.7'

        fields = header[2].split()
        assert fields[0] == 'FIELDS'
        assert 'x' in fields and 'y' in fields and 'z' in fields
        if self.use_intensity:
            assert 'intensity' in fields
        self.fields = fields[1:]

        sizes = header[3].split()
        assert sizes[0] == 'SIZE' and len(sizes) == len(fields)
        self.sizes = [int(size) for size in sizes[1:]]

        types = header[4].split()
        assert types[0] == 'TYPE' and len(types) == len(fields)
        self.types = types[1:]

        # convert to struct format
        self.formats = [''] * len(self.fields)
        for i in range(len(self.fields)):
            size = self.sizes[i]
            t = self.types[i]
            if size == 1:
                if t == 'I':
                    self.formats[i] = 'b'
                elif t == 'U':
                    self.formats[i] = 'B'
                else:
                    raise ValueError
            elif size == 2:
                if t == 'I':
                    self.formats[i] = 'h'
                elif t == 'U':
                    self.formats[i] = 'H'
                else:
                    raise ValueError
            elif size == 4:
                if t == 'I':
                    self.formats[i] = 'i'
                elif t == 'U':
                    self.formats[i] = 'I'
                elif t == 'F':
                    self.formats[i] = 'f'
                else:
                    raise ValueError
            elif size == 8:
                if t == 'I':
                    self.formats[i] = 'q'
                elif t == 'U':
                    self.formats[i] = 'Q'
                elif t == 'F':
                    self.formats[i] = 'd'
                else:
                    raise ValueError
            else:
                raise ValueError

        counts = header[5].split()
        assert counts[0] == 'COUNT' and len(counts) == len(fields)
        self.counts = [int(count) for count in counts[1:]]

        viewpoint = header[8].split()
        assert viewpoint[0] == 'VIEWPOINT'
        self.viewpoint = [int(v) for v in viewpoint[1:]]

        points = header[9].split()
        assert points[0] == 'POINTS'
        self.points = int(points[1])

        data = header[10].strip()
        assert data == 'DATA binary'

    def parse_binary_data(self, binary_data):
        output_points = []
        ptr = 0
        point_len = 4 if self.use_intensity else 3
        for p_idx in range(self.points):
            point = [0] * point_len
            for f_idx, field in enumerate(self.fields):
                cur_data_len = self.sizes[f_idx] * self.counts[f_idx]
                cur_data = binary_data[ptr:ptr + cur_data_len]
                ptr += cur_data_len
                # check field of interest
                idx = -1
                if field == 'x':
                    idx = 0
                elif field == 'y':
                    idx = 1
                elif field == 'z':
                    idx = 2
                elif field == 'intensity' and self.use_intensity:
                    idx = 3
                # assign converted value
                if idx != -1:
                    point[idx] = struct.unpack(self.formats[f_idx], cur_data)[0]
            output_points.append(point)
        output_points = np.array(output_points).astype(np.float32)
        return output_points

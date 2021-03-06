# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import Common.Grpc.fl_grpc_pb2 as fl__grpc__pb2


class FL_GrpcStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.UpdateIdx_uint32 = channel.unary_unary(
        '/FL_Grpc/UpdateIdx_uint32',
        request_serializer=fl__grpc__pb2.IdxRequest_uint32.SerializeToString,
        response_deserializer=fl__grpc__pb2.IdxResponse_uint32.FromString,
        )
    self.UpdateGrad_int32 = channel.unary_unary(
        '/FL_Grpc/UpdateGrad_int32',
        request_serializer=fl__grpc__pb2.GradRequest_int32.SerializeToString,
        response_deserializer=fl__grpc__pb2.GradResponse_int32.FromString,
        )
    self.UpdateGrad_float = channel.unary_unary(
        '/FL_Grpc/UpdateGrad_float',
        request_serializer=fl__grpc__pb2.GradRequest_float.SerializeToString,
        response_deserializer=fl__grpc__pb2.GradResponse_float.FromString,
        )
    self.DataTrans_int32 = channel.unary_unary(
        '/FL_Grpc/DataTrans_int32',
        request_serializer=fl__grpc__pb2.DataRequest_int32.SerializeToString,
        response_deserializer=fl__grpc__pb2.DataResponse_int32.FromString,
        )
    self.Update_SignSGD = channel.unary_unary(
        '/FL_Grpc/Update_SignSGD',
        request_serializer=fl__grpc__pb2.signSGD_Request.SerializeToString,
        response_deserializer=fl__grpc__pb2.signSGD_Response.FromString,
        )


class FL_GrpcServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def UpdateIdx_uint32(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def UpdateGrad_int32(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def UpdateGrad_float(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def DataTrans_int32(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Update_SignSGD(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_FL_GrpcServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'UpdateIdx_uint32': grpc.unary_unary_rpc_method_handler(
          servicer.UpdateIdx_uint32,
          request_deserializer=fl__grpc__pb2.IdxRequest_uint32.FromString,
          response_serializer=fl__grpc__pb2.IdxResponse_uint32.SerializeToString,
      ),
      'UpdateGrad_int32': grpc.unary_unary_rpc_method_handler(
          servicer.UpdateGrad_int32,
          request_deserializer=fl__grpc__pb2.GradRequest_int32.FromString,
          response_serializer=fl__grpc__pb2.GradResponse_int32.SerializeToString,
      ),
      'UpdateGrad_float': grpc.unary_unary_rpc_method_handler(
          servicer.UpdateGrad_float,
          request_deserializer=fl__grpc__pb2.GradRequest_float.FromString,
          response_serializer=fl__grpc__pb2.GradResponse_float.SerializeToString,
      ),
      'DataTrans_int32': grpc.unary_unary_rpc_method_handler(
          servicer.DataTrans_int32,
          request_deserializer=fl__grpc__pb2.DataRequest_int32.FromString,
          response_serializer=fl__grpc__pb2.DataResponse_int32.SerializeToString,
      ),
      'Update_SignSGD': grpc.unary_unary_rpc_method_handler(
          servicer.Update_SignSGD,
          request_deserializer=fl__grpc__pb2.signSGD_Request.FromString,
          response_serializer=fl__grpc__pb2.signSGD_Response.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'FL_Grpc', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))

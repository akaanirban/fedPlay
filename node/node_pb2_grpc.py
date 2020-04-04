# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import node_pb2 as node__pb2


class FederatedNodeStub(object):
    """Missing associated documentation comment in .proto file"""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.InitializeData = channel.unary_unary(
                '/FederatedNode/InitializeData',
                request_serializer=node__pb2.InitializeParams.SerializeToString,
                response_deserializer=node__pb2.Empty.FromString,
                )
        self.Train = channel.unary_unary(
                '/FederatedNode/Train',
                request_serializer=node__pb2.TrainParams.SerializeToString,
                response_deserializer=node__pb2.Model.FromString,
                )
        self.UpdateState = channel.unary_unary(
                '/FederatedNode/UpdateState',
                request_serializer=node__pb2.UpdateParams.SerializeToString,
                response_deserializer=node__pb2.Empty.FromString,
                )
        self.SendModel = channel.unary_unary(
                '/FederatedNode/SendModel',
                request_serializer=node__pb2.Model.SerializeToString,
                response_deserializer=node__pb2.Empty.FromString,
                )
        self.SendPredictions = channel.unary_unary(
                '/FederatedNode/SendPredictions',
                request_serializer=node__pb2.Empty.SerializeToString,
                response_deserializer=node__pb2.Predictions.FromString,
                )


class FederatedNodeServicer(object):
    """Missing associated documentation comment in .proto file"""

    def InitializeData(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Train(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateState(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendModel(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendPredictions(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FederatedNodeServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'InitializeData': grpc.unary_unary_rpc_method_handler(
                    servicer.InitializeData,
                    request_deserializer=node__pb2.InitializeParams.FromString,
                    response_serializer=node__pb2.Empty.SerializeToString,
            ),
            'Train': grpc.unary_unary_rpc_method_handler(
                    servicer.Train,
                    request_deserializer=node__pb2.TrainParams.FromString,
                    response_serializer=node__pb2.Model.SerializeToString,
            ),
            'UpdateState': grpc.unary_unary_rpc_method_handler(
                    servicer.UpdateState,
                    request_deserializer=node__pb2.UpdateParams.FromString,
                    response_serializer=node__pb2.Empty.SerializeToString,
            ),
            'SendModel': grpc.unary_unary_rpc_method_handler(
                    servicer.SendModel,
                    request_deserializer=node__pb2.Model.FromString,
                    response_serializer=node__pb2.Empty.SerializeToString,
            ),
            'SendPredictions': grpc.unary_unary_rpc_method_handler(
                    servicer.SendPredictions,
                    request_deserializer=node__pb2.Empty.FromString,
                    response_serializer=node__pb2.Predictions.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'FederatedNode', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class FederatedNode(object):
    """Missing associated documentation comment in .proto file"""

    @staticmethod
    def InitializeData(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/FederatedNode/InitializeData',
            node__pb2.InitializeParams.SerializeToString,
            node__pb2.Empty.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Train(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/FederatedNode/Train',
            node__pb2.TrainParams.SerializeToString,
            node__pb2.Model.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateState(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/FederatedNode/UpdateState',
            node__pb2.UpdateParams.SerializeToString,
            node__pb2.Empty.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SendModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/FederatedNode/SendModel',
            node__pb2.Model.SerializeToString,
            node__pb2.Empty.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SendPredictions(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/FederatedNode/SendPredictions',
            node__pb2.Empty.SerializeToString,
            node__pb2.Predictions.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)
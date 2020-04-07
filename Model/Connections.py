"""
The module contains a definition of the class of connections in a neuron
"""


class Connection:
    """
    Connection in a neuron
    """

    def __init__(self, node, weight):
        self.node = node
        self.weight = weight


class Connections:
    """
    Connections in a neuron
    """

    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        """
        Add connection to connection list
        :param connection: new connection -> Connection
        """
        self.connections.append(connection)

    def update_connection(self, old_node, new_node):
        """
        Update the node of connection
        :param old_node: node to be updated -> BaseNode
        :param new_node: new node -> BaseNode
        """
        try:
            connection = next(conn for conn in self.connections if conn.node.node_id == old_node.node_id)
        except StopIteration:
            self.add_connection(Connection(new_node, self.connections[0].weight))
            return
        connection.node = new_node

    def is_valid_connection(self, connection) -> bool:
        """
        Check for a connection in the connection list
        :param connection: verifiable connection -> Connection
        """
        conn_nodes = [conn.node.node_id for conn in self.connections]
        if connection.node.node_id in conn_nodes:
            return False
        return True

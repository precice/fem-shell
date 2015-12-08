/***********************************************\
 *                                              *
 *    meshGen - Mesh Generator for XDA meshes   *
 *                                              *
 *  by Stephan Herb -- Master Thesis, Dec 2015  *
 *                                              *
 \**********************************************/

#include <vector>
#include <stdlib.h>
#include <string>  // std::string, std::stoi <-- !!! requires -std=c++11 for g++
#include <iostream>  // std::ios, std::istream, std::cout
#include <fstream>   // std::filebuf

int main(int argc, char *argv[])
{
    // We need all parameters, otherwise the mesh cannot be created correctly
    if (argc != 14)
    {
        std::cout << "usage: " << argv[0] << " type nx ny min_x min_y max_x max_y bcids factor loading ul_lr dead-axis filename\n";
        std::cout << "type: Q|q for Quad-4, T|t for Tri-3\n"
                  << "nx: no. elements on primary axis\n"
                  << "ny: no. elements on secondary axis\n"
                  << "min_x: minimum position on the primary axis\n"
                  << "min_y: minimum position on the secondary axis\n"
                  << "max_x: maximum position on the primary axis\n"
                  << "max_y: maximum position on the secondary axis\n"
                  << "bcids: comma-separated list of boundary condition IDs in the ordering top,bottom,left,right border (e.g. 2,0,20,21). If border has no BC, type -1\n"
                  << "factor: global factor multiplied on all force entries in the force file\n"
                  << "loading: 0 for no loading at all, 1 for concentrated load on central node, 2 for uniform load on entire mesh\n"
                  << "ul_lr: 1 if triangle hypotenuse should face the lower right corner of the divided square, 0 for 90° rotated orientation. If type is Quad-4, the value doesn't matter.\n"
                  << "dead-axis: 'x','y' or 'z' to specify which axis should be considered dead. The resulting mesh will lie in the 'yz','xz' or 'xy'-plane, respectively.\n"
                  << "filename: the name of the mesh file to create\n";
        return -1;
    }

    char type = argv[1][0];

    // edit here, for more types in the future:
    if (type != 'Q' && type != 'q' &&
        type != 'T' && type != 't')
    {
        std::cout << "Invalid element type specified: \"" << type << "\". "
                  << "Only 'Q'|'q' and 'T'|'t' are allowed.\n";
        return -1;
    }
    // bring the type into lower case for better management
    // in the rest of the program:
    if (type == 'Q')
        type = 'q';
    else if (type == 'T')
        type = 't';

    int nx = atoi(argv[2]);
    if (nx <= 0)
    {
        std::cout << "Invalid number of elements on primary axis! Only positive integer values allowed.\n";
        return -1;
    }

    int ny = atoi(argv[3]);
    if (ny <= 0)
    {
        std::cout << "Invalid number of elements on secondary axis! Only positive integer values allowed.\n";
        return -1;
    }

    double min_x = atof(argv[4]);
    double min_y = atof(argv[5]);
    double max_x = atof(argv[6]);
    double max_y = atof(argv[7]);

    // get comma-separated list of boundary condition IDs
    // expected format: int,int,int,int
    std::string bcids = argv[8];
    std::size_t first = bcids.find_first_of(",");
    int curPos = 0;
    int t_bcid = -1;
    if (first != std::string::npos)
    {
        std::string str = bcids.substr(0, first);
        t_bcid = std::stoi(str);
    }
    curPos = first+1;
    first = bcids.find_first_of(",", first+1);
    int b_bcid = -1;
    if (first != std::string::npos)
    {
        std::string str = bcids.substr(curPos, first-curPos);
        b_bcid = std::stoi(str);
    }
    curPos = first+1;
    first = bcids.find_first_of(",", first+1);
    int l_bcid = -1;
    if (first != std::string::npos)
    {
        std::string str = bcids.substr(curPos, first-curPos);
        l_bcid = std::stoi(str);
    }
    curPos = first+1;
    std::string str = bcids.substr(curPos, bcids.size());
    int r_bcid = std::stoi(str);

    double factor = atof(argv[9]);

    int loading = atoi(argv[10]);

    bool ul_lr = atoi(argv[11])==1? true : false; // upper left to lower right diagonal, or ur_ll?

    char deadAxis = argv[12][0];

    char* fname = argv[13];

    bool bleft = false;
    if (l_bcid >= 0)
        bleft = true;
    bool bright = false;
    if (r_bcid >= 0)
        bright = true;
    bool btop = false;
    if (t_bcid >= 0)
        btop = true;
    bool bbottom = false;
    if (b_bcid >= 0)
        bbottom = true;

    if (deadAxis != 'x' && deadAxis != 'y' && deadAxis != 'z')
    {
        std::cout << "Invalid parameter for dead axis: \"" << deadAxis << "\". Only 'x','y' and 'z' are allowed.\n";
        return -1;
    }

    int n_elem = nx*ny;
    if (type == 't') // we split every rectangle into two triangles
        n_elem *= 2;
    int n_nodes = (nx+1)*(ny+1);

    std::vector<std::vector<double> > nodes;
    std::vector<std::vector<int> > elem;

    double fracx = (max_x-min_x)/(double)nx;
    double fracy = (max_y-min_y)/(double)ny;
    // creates nodes
    for (int y = 0; y <= ny; y++)
    {
        std::vector<double> node(3, 0.0);
        if (deadAxis == 'z') // secondary axis is y
            node[1] = min_y + y*fracy;
        else // deadAxis = y|x -> secondary axis is z
            node[2] = min_y + y*fracy;
        for (int x = 0; x <= nx; x++)
        {
            if (deadAxis == 'x') // primary axis is y
                node[1] = (min_x + x*fracx);
            else // deadAxis = y|z -> primary axis is x
                node[0] = (min_x + x*fracx);

            nodes.push_back(node);
        }
    }

    // create elements
    for (int y = 0; y < ny; y++)
    {
        if (type == 'q')
        {
            std::vector<int> quad(4);
            for (int x = 0; x < nx; x++)
            {
                /* 3-----2
                   |     |
                   |     |
                   0-----1 */
                int n_id = x + y*(nx+1);
                quad[0] = n_id;
                quad[1] = n_id + 1;
                quad[2] = n_id + (nx+1) + 1;
                quad[3] = n_id + (nx+1);

                elem.push_back(quad);
            }
        }
        else if (type == 't')
        {
            std::vector<int> tri1(3);
            std::vector<int> tri2(3);
            for (int x = 0; x < nx; x++)
            {
                int n_id = x + y*(nx+1);
                if (ul_lr)
                {
                    /* 2|2---1
                       |\\   |
                       | \\  |
                       |  \\ |
                       0---1|0 */
                    tri1[0] = n_id;
                    tri1[1] = n_id + 1;
                    tri1[2] = n_id + (nx+1);

                    tri2[0] = n_id + 1;
                    tri2[1] = n_id + (nx+1) + 1;
                    tri2[2] = n_id + (nx+1);
                }
                else
                {
                    /* 2---0|1
                       |   //|
                       |  // |
                       | //  |
                       1|0---2 */
                    tri1[0] = n_id;
                    tri1[1] = n_id + (nx+1) + 1;
                    tri1[2] = n_id + 1;

                    tri2[0] = n_id + (nx+1) + 1;
                    tri2[1] = n_id;
                    tri2[2] = n_id + (nx+1);
                }
                elem.push_back(tri1);
                elem.push_back(tri2);
            }
        }
    }

    std::string meshname = fname;
    meshname += ".xda";
    std::filebuf fb;
    fb.open (meshname.c_str(), std::ios::out);
    std::ostream os(&fb);

    // The header of the XDA mesh file:
    os << "libMesh-0.7.0+\n";
    os << n_elem  << "      # number of elements\n";
    os << n_nodes << "      # number of nodes\n";
    os << ".        # boundary condition specification file\n";
    os << "n/a      # subdomain id specification file\n";
    os << "n/a      # processor id specification file\n";
    os << "n/a      # p-level specification file\n";
    os << n_elem << "      # n_elem at level 0, [ type (n0 ... nN-1) ]\n";

    // write the elements:
    char elType = 't';
    if (type == 't')
        elType = '3';
    else if (type == 'q')
        elType = '5';
    for (unsigned int i = 0; i < elem.size(); i++)
    {
        std::vector<int> cur_elem = elem[i];
        os << elType;
        for (unsigned int k = 0; k < cur_elem.size(); k++)
            os << " " << cur_elem[k];
        os << "\n";
    }

    // write the node coordinates:
    for (unsigned int i = 0; i < nodes.size(); i++)
    {
        std::vector<double> cur_node = nodes[i];
        os << cur_node[0] << " " << cur_node[1] << " " << cur_node[2] << "\n";
    }

    // calculate the number of edges with boundary conditions:
    int noBC = 0;
    if (bleft)
        noBC += ny;
    if (bright)
        noBC += ny;
    if (btop)
        noBC += nx;
    if (bbottom)
        noBC += nx;
    os << noBC << "        # number of boundary conditions\n";
    // boundary condition format:
    // x y z
    // x: ID of element
    // y: number of edge of element.
    //    edge 0 from vertex 0 to 1,
    //    edge 1 from vertex 1 to 2, etc.
    // z: BC ID

    // top and bottom borders:
    for (int i = 0; i < nx; i++)
    {
        if (type == 't')
        {
            if (ul_lr)
            {
                if (bbottom)
                    os << 2*i << " 0 " << b_bcid << "\n"; // bottom border
                if (btop)
                    os << 2*nx*ny-2*i-1 << " 1 " << t_bcid << "\n"; // top border
            }
            else
            {
                if (bbottom)
                    os << 2*i << " 2 " << b_bcid << "\n"; // bottom border
                if (btop)
                    os << 2*nx*ny-2*i-1 << " 2 " << t_bcid << "\n"; // top border
            }
        }
        else if (type == 'q')
        {
            if (bbottom)
                os << i << " 0 " << b_bcid << "\n"; // bottom border
            if (btop)
                os << nx*ny-1-i << " 2 " << t_bcid << "\n"; // top border
        }
    }
    // left and right borders:
    for (int i = 0; i < ny; i++)
    {
        if (type == 't')
        {
            if (ul_lr)
            {
                if (bleft)
                    os << 2*nx*i << " 2 " << l_bcid << "\n"; // left border
                if (bright)
                    os << 2*nx*(i+1)-1 << " 0 " << r_bcid << "\n"; // right border
            }
            else
            {
                if (bleft)
                    os << 2*nx*i+1 << " 1 " << l_bcid << "\n"; // left border
                if (bright)
                    os << 2*nx*(i+1)-2 << " 1 " << r_bcid << "\n"; // right border
            }
        }
        else if (type == 'q')
        {
            if (bleft)
                os << nx*i << " 3 " << l_bcid << "\n"; // left border
            if (bright)
                os << nx*(i+1)-1 << " 1 " << r_bcid << "\n"; // right border
        }
    }
    fb.close();

    if (loading <= 0) // if no loading is desired, we are done.
        return 0;
    std::string forcename = fname;
    forcename += "_f"; // convention: force file is named like mesh file with "_f" at the end
    fb.open (forcename.c_str(), std::ios::out);
    std::ostream os2(&fb);

    os2 << n_nodes << "\n";
    if (loading == 1) // concentrated loading at central node
    {
        os2 << factor << "\n";
        for (unsigned int i = 0; i < nodes.size()-1; i++)
        {
            // force is applied perpendicular to mesh plane (for plate testing)
            // direction of force could be also command-line argument (for the future)
            if (i == (unsigned)n_nodes/2)
                if (deadAxis == 'x')
                    os2 << "1 0 0 0 0 0\n";
                else if (deadAxis == 'y')
                    os2 << "0 1 0 0 0 0\n";
                else
                    os2 << "0 0 1 0 0 0\n";
            else
                os2 << "0 0 0 0 0 0\n";
        }
    }
    else if (loading == 2) // uniformly distributed loading
    {
        // convert area force to nodal force:
        // factor * elem_x_len * elem_y_len / #nodes_of_elem * #neighboring_elements_sharing_this_node
        // for quad's: f*xlen*ylen / 4 * 4 -> f*xlen*ylen
        // for tri's: f*xlen*ylen/2 / 3 * 6 -> f*xlen*ylen
        os2 << (factor*((max_x-min_x)/(double)nx)*((max_y-min_y)/(double)ny)) << "\n";
        // force is applied perpendicular to mesh plane (for plate testing)
        // direction of force could be also command-line argument (for the future)
        if (deadAxis == 'x')
            for (unsigned int i = 0; i < nodes.size()-1; i++)
                os2 << "1 0 0 0 0 0\n";
        else if (deadAxis == 'y')
            for (unsigned int i = 0; i < nodes.size()-1; i++)
                os2 << "0 1 0 0 0 0\n";
        else
            for (unsigned int i = 0; i < nodes.size()-1; i++)
                os2 << "0 0 1 0 0 0\n";

    }
    fb.close();

    return 0;
}